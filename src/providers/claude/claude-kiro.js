import { atomicWriteFile, withFileLock } from '../../utils/file-lock.js';
import axios from 'axios';
import logger from '../../utils/logger.js';
import { v4 as uuidv4 } from 'uuid';
import { promises as fs } from 'fs';
import * as path from 'path';
import * as os from 'os';
import * as crypto from 'crypto';
import * as http from 'http';
import * as https from 'https';
import { getProviderModels } from '../provider-models.js';
import { 
    countTextTokens as countTextTokensUtil, 
    estimateInputTokens as estimateInputTokensUtil, 
    countTokensAnthropic as countTokensUtil,
    processContent as processContentUtil,
    getContentText as getContentTextUtil
} from '../../utils/token-utils.js';
import { configureAxiosProxy, configureTLSSidecar, isTLSSidecarEnabledForProvider } from '../../utils/proxy-utils.js';
import { isRetryableNetworkError, MODEL_PROVIDER, formatExpiryLog } from '../../utils/common.js';
import { getProviderPoolManager } from '../../services/service-manager.js';

const KIRO_THINKING = {
    MIN_BUDGET_TOKENS: 1024,
    MAX_BUDGET_TOKENS: 24576,
    DEFAULT_BUDGET_TOKENS: 20000,
    START_TAG: '<thinking>',
    END_TAG: '</thinking>',
    MODE_TAG: '<thinking_mode>',
    MAX_LEN_TAG: '<max_thinking_length>',
};

const KIRO_CONSTANTS = {
    REFRESH_URL: 'https://prod.{{region}}.auth.desktop.kiro.dev/refreshToken',
    REFRESH_IDC_URL: 'https://oidc.{{region}}.amazonaws.com/token',
    BASE_URL: 'https://q.{{region}}.amazonaws.com/generateAssistantResponse',
    DEFAULT_REGION: 'us-east-1',
    DEFAULT_MODEL_NAME: 'claude-sonnet-4-5',
    AXIOS_TIMEOUT: 300000, // 5 minutes timeout for long-running requests
    STREAM_TIMEOUT: 0, // Disable axios timeout for real streaming requests.
    TOKEN_REFRESH_TIMEOUT: 15000, // 15 seconds timeout for token refresh (shorter to avoid blocking)
    USER_AGENT: 'KiroIDE',
    KIRO_VERSION: process.env.KIRO_VERSION || '1.0.6',
    CONTENT_TYPE_JSON: 'application/json',
    ACCEPT_JSON: 'application/json',
    AUTH_METHOD_SOCIAL: 'social',
    CHAT_TRIGGER_TYPE_MANUAL: 'MANUAL',
    ORIGIN_AI_EDITOR: 'AI_EDITOR',
    TOTAL_CONTEXT_TOKENS: 200000, // Claude Sonnet 4.5 actual context is 200K
};

const KIRO_MAX_TOOL_NAME_LENGTH = 64;
let kiroThrottleQueue = Promise.resolve();
let kiroLastRequestStartedAt = 0;

// Lazy-loaded PDF parser - only loaded when needed to avoid startup overhead
let _pdfParseModule = null;
async function getPdfParser() {
    if (_pdfParseModule === null) {
        try {
            const mod = await import('pdf-parse');
            _pdfParseModule = mod.default || mod.PDFParse || mod;
        } catch (err) {
            logger.warn(`[Kiro] pdf-parse module not available: ${err.message}`);
            _pdfParseModule = false;
        }
    }
    return _pdfParseModule || null;
}

/**
 * Parse a base64-encoded PDF into plain text.
 * Returns null on failure.
 */
async function parsePdfBase64ToText(base64Data) {
    try {
        const pdfParse = await getPdfParser();
        if (!pdfParse) return null;
        const buffer = Buffer.from(base64Data, 'base64');
        if (typeof pdfParse === 'function' && !pdfParse.prototype?.getText) {
            const result = await pdfParse(buffer);
            return result && typeof result.text === 'string' ? result.text : null;
        }

        const parser = new pdfParse({ data: buffer });
        const result = await parser.getText();
        await parser.destroy();
        return result && typeof result.text === 'string' ? result.text : null;
    } catch (err) {
        logger.warn(`[Kiro] Failed to parse PDF: ${err.message}`);
        return null;
    }
}

/**
 * Generate an Anthropic-style message ID: "msg_01" + 22 random base32 characters.
 * Real Claude API responses use this format (e.g., "msg_01ABC123...").
 */
function generateAnthropicMessageId() {
    const base32Chars = 'ABCDEFGHJKMNPQRSTVWXYZ0123456789'; // Crockford-style base32
    const randomBytes = crypto.randomBytes(22);
    let id = 'msg_01';
    for (let i = 0; i < 22; i++) {
        id += base32Chars[randomBytes[i] % base32Chars.length];
    }
    return id;
}

/**
 * Generate an Anthropic-style request ID: "req_01" + 22 random base32 characters.
 */
function generateAnthropicRequestId() {
    const base32Chars = 'ABCDEFGHJKMNPQRSTVWXYZ0123456789';
    const randomBytes = crypto.randomBytes(22);
    let id = 'req_01';
    for (let i = 0; i < 22; i++) {
        id += base32Chars[randomBytes[i] % base32Chars.length];
    }
    return id;
}

function shortenKiroToolName(name) {
    const rawName = String(name || '');
    if (rawName.length <= KIRO_MAX_TOOL_NAME_LENGTH) {
        return rawName;
    }

    const hash = crypto.createHash('sha256').update(rawName).digest('hex').slice(0, 12);
    const prefixLength = KIRO_MAX_TOOL_NAME_LENGTH - hash.length - 1;
    return `${rawName.slice(0, prefixLength)}_${hash}`;
}

function buildKiroToolNameMaps(tools) {
    const aliasToOriginal = new Map();
    const originalToAlias = new Map();

    if (Array.isArray(tools)) {
        for (const tool of tools) {
            const originalName = tool?.name;
            if (!originalName) continue;
            const aliasName = shortenKiroToolName(originalName);
            originalToAlias.set(originalName, aliasName);
            if (aliasName !== originalName) {
                aliasToOriginal.set(aliasName, originalName);
            }
        }
    }

    return {
        aliasToOriginal,
        toKiroName: (name) => originalToAlias.get(name) || shortenKiroToolName(name),
        fromKiroName: (name) => aliasToOriginal.get(name) || name
    };
}

function restoreKiroToolCallNames(toolCalls, toolNameMaps) {
    if (!toolCalls || !toolNameMaps?.fromKiroName) {
        return toolCalls;
    }

    return toolCalls.map(toolCall => ({
        ...toolCall,
        function: {
            ...toolCall.function,
            name: toolNameMaps.fromKiroName(toolCall.function?.name)
        }
    }));
}

function toOpenAIToolCall(toolCall) {
    if (!toolCall) return null;
    if (toolCall.function?.name !== undefined) return toolCall;

    const id = toolCall.id || toolCall.toolUseId || `call_${uuidv4().replace(/-/g, '').substring(0, 8)}`;
    let args = toolCall.input;
    if (args === undefined || args === null) {
        args = '{}';
    } else if (typeof args !== 'string') {
        try {
            args = JSON.stringify(args);
        } catch {
            args = String(args);
        }
    }

    return {
        id,
        type: 'function',
        function: {
            name: toolCall.name || toolCall.function?.name || 'unknown_tool',
            arguments: args
        }
    };
}

function toOpenAIToolCalls(toolCalls) {
    if (!Array.isArray(toolCalls)) return [];
    return toolCalls.map(toOpenAIToolCall).filter(Boolean);
}

function normalizeToolCallForStream(toolCall) {
    if (!toolCall) return null;
    if (toolCall.toolUseId && toolCall.name !== undefined) {
        return {
            toolUseId: toolCall.toolUseId,
            name: toolCall.name,
            input: normalizeToolCallArguments(toolCall.input || {})
        };
    }

    return {
        toolUseId: toolCall.id || `tool_${uuidv4()}`,
        name: toolCall.function?.name || toolCall.name || 'unknown_tool',
        input: normalizeToolCallArguments(toolCall.function?.arguments || toolCall.input || '{}')
    };
}

/**
 * Diagnose why a tool call's JSON input is invalid (borrowed from kiro-gateway).
 * Returns a diagnostic string, or null if the JSON is valid or empty.
 */
function diagnoseJsonTruncation(raw) {
    if (!raw || typeof raw !== 'string') return null;
    const stripped = raw.trim();
    if (!stripped) return null;
    try { JSON.parse(stripped); return null; } catch (e) { /* proceed to diagnose */ }

    const openBraces = (stripped.match(/{/g) || []).length;
    const closeBraces = (stripped.match(/}/g) || []).length;
    const openBrackets = (stripped.match(/\[/g) || []).length;
    const closeBrackets = (stripped.match(/]/g) || []).length;

    if (stripped.startsWith('{') && !stripped.endsWith('}')) {
        return `missing ${openBraces - closeBraces} closing brace(s) (${stripped.length} bytes)`;
    }
    if (stripped.startsWith('[') && !stripped.endsWith(']')) {
        return `missing ${openBrackets - closeBrackets} closing bracket(s) (${stripped.length} bytes)`;
    }
    if (openBraces !== closeBraces) {
        return `unbalanced braces (${openBraces} open, ${closeBraces} close, ${stripped.length} bytes)`;
    }
    if (openBrackets !== closeBrackets) {
        return `unbalanced brackets (${openBrackets} open, ${closeBrackets} close, ${stripped.length} bytes)`;
    }
    return `invalid JSON (${stripped.length} bytes)`;
}

function getKiroRequestMinIntervalMs(config) {
    const value = Number(config?.KIRO_REQUEST_MIN_INTERVAL_MS);
    return Number.isFinite(value) && value > 0 ? Math.floor(value) : 0;
}

async function acquireKiroRequestSlot(config) {
    const minIntervalMs = getKiroRequestMinIntervalMs(config);
    if (minIntervalMs <= 0) {
        return () => {};
    }

    let releaseCurrent;
    const previous = kiroThrottleQueue.catch(() => {});
    kiroThrottleQueue = previous.then(() => new Promise(resolve => {
        releaseCurrent = resolve;
    }));

    await previous;

    const elapsedMs = Date.now() - kiroLastRequestStartedAt;
    const waitMs = Math.max(0, minIntervalMs - elapsedMs);
    if (waitMs > 0) {
        await new Promise(resolve => setTimeout(resolve, waitMs));
    }
    kiroLastRequestStartedAt = Date.now();

    let released = false;
    return () => {
        if (released) return;
        released = true;
        releaseCurrent();
    };
}

function normalizeKiroToolInput(input) {
    if (input === undefined || input === null) {
        return '';
    }
    if (typeof input === 'string') {
        return input;
    }
    if (typeof input === 'object') {
        try {
            return JSON.stringify(input);
        } catch (e) {
            return String(input);
        }
    }
    return String(input);
}

const AWS_EVENT_STREAM_MIN_MESSAGE_SIZE = 16;
const AWS_EVENT_STREAM_PRELUDE_SIZE = 12;
const AWS_EVENT_STREAM_MAX_MESSAGE_SIZE = 16 * 1024 * 1024;
const AWS_HEADER_VALUE_TYPE = {
    BOOL_TRUE: 0,
    BOOL_FALSE: 1,
    BYTE: 2,
    SHORT: 3,
    INTEGER: 4,
    LONG: 5,
    BYTE_ARRAY: 6,
    STRING: 7,
    TIMESTAMP: 8,
    UUID: 9,
};

function toBuffer(value) {
    if (Buffer.isBuffer(value)) return value;
    if (value instanceof ArrayBuffer) return Buffer.from(value);
    if (ArrayBuffer.isView(value)) {
        return Buffer.from(value.buffer, value.byteOffset, value.byteLength);
    }
    return Buffer.from(String(value || ''), 'utf8');
}

function getKiroEventErrorMessage(event) {
    const data = event?.data;
    const rawError = data?.error;
    if (!data && !rawError) {
        return 'Kiro upstream event-stream error';
    }
    if (typeof rawError === 'string') {
        return rawError;
    }
    if (rawError && typeof rawError === 'object') {
        return rawError.message || rawError.errorMessage || rawError.error || JSON.stringify(rawError);
    }
    return data?.eventType || data?.messageType || 'Kiro upstream event-stream error';
}

function createKiroEventStreamError(event) {
    const message = getKiroEventErrorMessage(event);
    const error = new Error(sanitizeProviderLeakText(message));
    error.kiroEventStreamError = true;
    error.kiroEventStreamData = event?.data;
    return error;
}

function isPlausibleAwsEventStreamPrelude(buffer, offset = 0) {
    if (!Buffer.isBuffer(buffer) || buffer.length - offset < AWS_EVENT_STREAM_PRELUDE_SIZE) {
        return false;
    }

    const totalLength = buffer.readUInt32BE(offset);
    const headersLength = buffer.readUInt32BE(offset + 4);
    return totalLength >= AWS_EVENT_STREAM_MIN_MESSAGE_SIZE &&
        totalLength <= AWS_EVENT_STREAM_MAX_MESSAGE_SIZE &&
        headersLength <= totalLength - AWS_EVENT_STREAM_MIN_MESSAGE_SIZE;
}

function readAwsHeaderValue(buffer, offset, type) {
    switch (type) {
        case AWS_HEADER_VALUE_TYPE.BOOL_TRUE:
            return { value: true, nextOffset: offset };
        case AWS_HEADER_VALUE_TYPE.BOOL_FALSE:
            return { value: false, nextOffset: offset };
        case AWS_HEADER_VALUE_TYPE.BYTE:
            if (offset + 1 > buffer.length) throw new Error('Incomplete byte header value');
            return { value: buffer.readInt8(offset), nextOffset: offset + 1 };
        case AWS_HEADER_VALUE_TYPE.SHORT:
            if (offset + 2 > buffer.length) throw new Error('Incomplete short header value');
            return { value: buffer.readInt16BE(offset), nextOffset: offset + 2 };
        case AWS_HEADER_VALUE_TYPE.INTEGER:
            if (offset + 4 > buffer.length) throw new Error('Incomplete integer header value');
            return { value: buffer.readInt32BE(offset), nextOffset: offset + 4 };
        case AWS_HEADER_VALUE_TYPE.LONG:
        case AWS_HEADER_VALUE_TYPE.TIMESTAMP:
            if (offset + 8 > buffer.length) throw new Error('Incomplete long header value');
            return { value: Number(buffer.readBigInt64BE(offset)), nextOffset: offset + 8 };
        case AWS_HEADER_VALUE_TYPE.BYTE_ARRAY:
        case AWS_HEADER_VALUE_TYPE.STRING: {
            if (offset + 2 > buffer.length) throw new Error('Incomplete variable header length');
            const length = buffer.readUInt16BE(offset);
            const valueStart = offset + 2;
            const valueEnd = valueStart + length;
            if (valueEnd > buffer.length) throw new Error('Incomplete variable header value');
            const raw = buffer.subarray(valueStart, valueEnd);
            return {
                value: type === AWS_HEADER_VALUE_TYPE.STRING ? raw.toString('utf8') : Buffer.from(raw),
                nextOffset: valueEnd
            };
        }
        case AWS_HEADER_VALUE_TYPE.UUID:
            if (offset + 16 > buffer.length) throw new Error('Incomplete uuid header value');
            return { value: buffer.subarray(offset, offset + 16).toString('hex'), nextOffset: offset + 16 };
        default:
            throw new Error(`Unsupported AWS event-stream header type: ${type}`);
    }
}

function parseAwsEventStreamHeaders(headersBuffer) {
    const headers = {};
    let offset = 0;

    while (offset < headersBuffer.length) {
        const nameLength = headersBuffer[offset];
        offset += 1;
        if (!nameLength) throw new Error('Invalid AWS event-stream header name length');
        if (offset + nameLength > headersBuffer.length) throw new Error('Incomplete AWS event-stream header name');

        const name = headersBuffer.subarray(offset, offset + nameLength).toString('utf8');
        offset += nameLength;
        if (offset >= headersBuffer.length) throw new Error('Missing AWS event-stream header type');

        const type = headersBuffer[offset];
        offset += 1;
        const parsed = readAwsHeaderValue(headersBuffer, offset, type);
        headers[name] = parsed.value;
        offset = parsed.nextOffset;
    }

    return headers;
}

function normalizeKiroParsedEvent(parsed, eventType = null, messageType = 'event') {
    const events = [];
    if (!parsed || typeof parsed !== 'object') {
        return events;
    }

    if (messageType === 'error' || messageType === 'exception') {
        events.push({
            type: 'error',
            data: {
                messageType,
                eventType,
                error: parsed
            }
        });
        return events;
    }

    if ((eventType === 'assistantResponseEvent' || parsed.content !== undefined) && !parsed.followupPrompt) {
        if (parsed.content !== undefined) {
            events.push({ type: 'content', data: parsed.content });
        }
    } else if (eventType === 'toolUseEvent' || parsed.name || parsed.toolUseId || parsed.input !== undefined) {
        if (parsed.name && parsed.toolUseId) {
            events.push({
                type: 'toolUse',
                data: {
                    name: parsed.name,
                    toolUseId: parsed.toolUseId,
                    input: normalizeKiroToolInput(parsed.input),
                    stop: parsed.stop || false
                }
            });
        } else if (parsed.input !== undefined) {
            events.push({
                type: 'toolUseInput',
                data: {
                    toolUseId: parsed.toolUseId,
                    input: normalizeKiroToolInput(parsed.input)
                }
            });
            if (parsed.stop === true) {
                events.push({
                    type: 'toolUseStop',
                    data: {
                        toolUseId: parsed.toolUseId,
                        stop: true
                    }
                });
            }
        } else if (parsed.stop === true) {
            events.push({
                type: 'toolUseStop',
                data: {
                    toolUseId: parsed.toolUseId,
                    stop: true
                }
            });
        }
    } else if (eventType === 'contextUsageEvent' || parsed.contextUsagePercentage !== undefined) {
        if (parsed.contextUsagePercentage !== undefined) {
            events.push({
                type: 'contextUsage',
                data: {
                    contextUsagePercentage: parsed.contextUsagePercentage
                }
            });
        }
    }

    return events;
}

function parseAwsEventStreamPayload(payloadBuffer, headers) {
    const messageType = headers[':message-type'] || 'event';
    const eventType = headers[':event-type'] || null;
    const payloadText = payloadBuffer.toString('utf8');

    if (!payloadText.trim()) {
        return [];
    }

    if (messageType === 'error' || messageType === 'exception') {
        let errorPayload = payloadText;
        try {
            errorPayload = JSON.parse(payloadText);
        } catch {}
        return [{
            type: 'error',
            data: {
                messageType,
                eventType: eventType || headers[':exception-type'] || headers[':error-code'] || null,
                error: errorPayload
            }
        }];
    }

    try {
        return normalizeKiroParsedEvent(JSON.parse(payloadText), eventType, messageType);
    } catch (error) {
        logger.warn(`[Kiro] Failed to parse AWS event-stream payload JSON (${eventType || messageType}): ${error.message}`);
        return [];
    }
}

function parseAwsEventStreamFrames(buffer, { recover = false } = {}) {
    const source = toBuffer(buffer);
    const events = [];
    let offset = 0;
    let recognized = false;

    while (offset < source.length) {
        if (source.length - offset < AWS_EVENT_STREAM_PRELUDE_SIZE) {
            break;
        }

        const totalLength = source.readUInt32BE(offset);
        const headersLength = source.readUInt32BE(offset + 4);
        const validPrelude = totalLength >= AWS_EVENT_STREAM_MIN_MESSAGE_SIZE &&
            totalLength <= AWS_EVENT_STREAM_MAX_MESSAGE_SIZE &&
            headersLength <= totalLength - AWS_EVENT_STREAM_MIN_MESSAGE_SIZE;

        if (!validPrelude) {
            if (!recover) {
                break;
            }
            offset += 1;
            continue;
        }

        recognized = true;
        if (source.length - offset < totalLength) {
            break;
        }

        const headersStart = offset + AWS_EVENT_STREAM_PRELUDE_SIZE;
        const headersEnd = headersStart + headersLength;
        const payloadEnd = offset + totalLength - 4;

        try {
            const headers = parseAwsEventStreamHeaders(source.subarray(headersStart, headersEnd));
            const payload = source.subarray(headersEnd, payloadEnd);
            events.push(...parseAwsEventStreamPayload(payload, headers));
            offset += totalLength;
        } catch (error) {
            logger.warn(`[Kiro] Failed to decode AWS event-stream frame: ${error.message}`);
            if (!recover) {
                offset += totalLength;
            } else {
                offset += 1;
            }
        }
    }

    return {
        events,
        remaining: source.subarray(offset),
        recognized
    };
}

class KiroAwsEventStreamDecoder {
    constructor() {
        this.buffer = Buffer.alloc(0);
    }

    feed(chunk) {
        const incoming = toBuffer(chunk);
        this.buffer = this.buffer.length > 0 ? Buffer.concat([this.buffer, incoming]) : incoming;
        if (this.buffer.length > AWS_EVENT_STREAM_MAX_MESSAGE_SIZE) {
            throw new Error(`AWS event-stream buffer overflow (${this.buffer.length} bytes)`);
        }

        if (this.buffer.length >= AWS_EVENT_STREAM_PRELUDE_SIZE &&
            !isPlausibleAwsEventStreamPrelude(this.buffer, 0)) {
            return { events: [], remaining: this.buffer, recognized: false };
        }

        const parsed = parseAwsEventStreamFrames(this.buffer, { recover: false });
        this.buffer = parsed.remaining;
        return parsed;
    }

    finish() {
        if (!this.buffer.length) {
            return { events: [], remaining: Buffer.alloc(0), recognized: false };
        }
        if (!isPlausibleAwsEventStreamPrelude(this.buffer, 0)) {
            const remaining = this.buffer;
            this.buffer = Buffer.alloc(0);
            return { events: [], remaining, recognized: false };
        }

        const parsed = parseAwsEventStreamFrames(this.buffer, { recover: false });
        this.buffer = Buffer.alloc(0);
        return parsed;
    }
}

// Per-model context window sizes for accurate token estimation
const MODEL_CONTEXT_TOKENS = {
    "claude-opus-4-7": 1000000,
    "claude-opus-4-6": 1000000,
    "claude-opus-4-5": 1000000,
    "claude-opus-4-5-20251101": 1000000,
    "claude-sonnet-4-6": 200000,
    "claude-sonnet-4-5": 200000,
    "claude-sonnet-4-5-20250929": 200000,
    "claude-sonnet-4-20250514": 200000,
    "claude-3-7-sonnet-20250219": 200000,
    "claude-haiku-4-5": 200000,
    "claude-haiku-4-5-20251001": 200000,
};

function normalizeContextLength(value) {
    if (value === undefined || value === null || value === '') {
        return null;
    }

    const numericValue = Number(value);
    return Number.isFinite(numericValue) && numericValue > 0 ? numericValue : null;
}

function findCustomModelConfigForModel(model, config = {}) {
    const targetModel = typeof model === 'string'
        ? model.replace(/^[^:]+:/, '')
        : '';
    if (!targetModel) {
        return null;
    }

    const customModels = Array.isArray(config?.customModels) ? config.customModels : [];
    return customModels.find(({ id, alias, actualModel } = {}) =>
        id === targetModel || alias === targetModel || actualModel === targetModel
    ) || null;
}

function getContextTokensForModel(model, config = {}, fallbackModel = null) {
    const customModelConfig = findCustomModelConfigForModel(model, config) ||
        findCustomModelConfigForModel(fallbackModel, config);
    const configuredModelContextLength = normalizeContextLength(customModelConfig?.contextLength);
    if (configuredModelContextLength !== null) {
        return configuredModelContextLength;
    }

    return MODEL_CONTEXT_TOKENS[model] || MODEL_CONTEXT_TOKENS[fallbackModel] || KIRO_CONSTANTS.TOTAL_CONTEXT_TOKENS;
}
// 从 provider-models.js 获取支持的模型列表
const KIRO_MODELS = getProviderModels(MODEL_PROVIDER.KIRO_API);

// 完整的模型映射表
const FULL_MODEL_MAPPING = {
    "claude-haiku-4-5":"claude-haiku-4.5",
    "claude-haiku-4-5-20251001":"claude-haiku-4.5",
    "claude-opus-4-7":"claude-opus-4.7",
    "claude-opus-4-6":"claude-opus-4.6",
    "claude-sonnet-4-6":"claude-sonnet-4.6",
    "claude-opus-4-5":"claude-opus-4.5",
    "claude-opus-4-5-20251101":"claude-opus-4.5",
    "claude-sonnet-4-5": "claude-sonnet-4.5",
    "claude-sonnet-4-5-20250929": "claude-sonnet-4.5",
    "claude-sonnet-4-20250514": "claude-sonnet-4.5",
    "claude-3-7-sonnet-20250219": "claude-sonnet-4.5"
};

function resolveKiroModel(model) {
    return FULL_MODEL_MAPPING[model] || model;
}

function isInvalidKiroModelError(error) {
    const data = error?.response?.data;
    if (data?.reason === 'INVALID_MODEL_ID') return true;
    if (Buffer.isBuffer(data) || data instanceof ArrayBuffer || ArrayBuffer.isView(data)) {
        const text = toBuffer(data).toString('utf8');
        return text.includes('INVALID_MODEL_ID') || text.includes('Invalid model ID');
    }
    if (typeof data === 'string') {
        return data.includes('INVALID_MODEL_ID') || data.includes('Invalid model ID');
    }
    return false;
}

function buildKiroToolsContext(tools, toolNameMaps) {
    if (!Array.isArray(tools) || tools.length === 0) {
        return {};
    }

    const kiroTools = tools.map(tool => ({
        toolSpecification: {
            name: toolNameMaps.toKiroName(tool.name),
            description: tool.description || '',
            inputSchema: {
                json: tool.input_schema || {}
            }
        }
    }));

    return { tools: kiroTools };
}

const KIRO_AUTH_TOKEN_FILE = "kiro-auth-token.json";

/**
 * Kiro API Service - Node.js implementation based on the Python ki2api
 * Provides OpenAI-compatible API for Claude Sonnet 4 via Kiro/CodeWhisperer
 */

/**
 * 根据当前配置生成唯一的机器码（Machine ID）
 * 确保每个配置对应一个唯一且不变的 ID
 * @param {Object} credentials - 当前凭证信息
 * @returns {string} SHA256 格式的机器码
 */
function generateMachineIdFromConfig(credentials) {
    // Combine credential identity with host-specific info to produce a unique
    // per-credential-per-machine ID. This prevents multiple pool entries sharing
    // the same clientId from producing identical machine fingerprints.
    const uniqueKey = credentials.uuid || credentials.profileArn || credentials.clientId || "KIRO_DEFAULT_MACHINE";
    const hostSalt = `${os.hostname()}-${os.userInfo().username}-${os.arch()}`;
    return crypto.createHash('sha256').update(`${uniqueKey}:${hostSalt}`).digest('hex');
}

/**
 * Generate system runtime info for User-Agent strings.
 * Mimics a real Kiro IDE (Electron-based) rather than a raw Node.js process.
 * @returns {Object} osName and electronVersion for UA construction
 */
function getSystemRuntimeInfo(config = {}) {
    const osPlatform = os.platform();
    const osRelease = os.release();
    // Kiro IDE runs on Electron — report Electron version, not Node.js
    const electronVersion = config.KIRO_ELECTRON_VERSION || process.env.KIRO_ELECTRON_VERSION || '33.4.0';

    // Kiro IDE's Electron reports a simplified OS version.
    // Real Electron os.release() on Windows gives major.minor.build but Kiro UA
    // typically uses a shorter format.
    let osName = osPlatform;
    if (osPlatform === 'win32') {
        // Extract just major.minor (e.g., "10.0" from "10.0.26100")
        const parts = osRelease.split('.');
        osName = `windows#${parts[0]}.${parts[1] || '0'}.0`;
    } else if (osPlatform === 'darwin') {
        osName = `macos#${osRelease}`;
    } else {
        osName = `${osPlatform}#${osRelease}`;
    }

    return {
        osName,
        nodeVersion: electronVersion  // Used as "md/electron#X.Y.Z" in UA
    };
}

// Helper functions for tool calls and JSON parsing

function isQuoteCharAt(text, index) {
    if (index < 0 || index >= text.length) return false;
    const ch = text[index];
    return ch === '"' || ch === "'" || ch === '`';
}

function findRealTag(text, tag, startIndex = 0) {
    let searchStart = Math.max(0, startIndex);
    while (true) {
        const pos = text.indexOf(tag, searchStart);
        if (pos === -1) return -1;
        
        const hasQuoteBefore = isQuoteCharAt(text, pos - 1);
        const hasQuoteAfter = isQuoteCharAt(text, pos + tag.length);
        if (!hasQuoteBefore && !hasQuoteAfter) {
            return pos;
        }
        
        searchStart = pos + 1;
    }
}

function isWhitespaceOnly(text) {
    if (text === null || text === undefined) return true;
    return String(text).trim().length === 0;
}

function generateFakeThinkingSignature() {
    return crypto.randomBytes(64).toString('base64');
}

/**
 * Find a "real" thinking end tag that is not quoted/backticked and is followed by '\n\n'.
 * This avoids prematurely closing a thinking block when the model mentions `</thinking>`
 * inside the thinking content.
 */
function findRealThinkingEndTag(buffer, startIndex = 0) {
    let searchStart = Math.max(0, startIndex);
    while (true) {
        const pos = findRealTag(buffer, KIRO_THINKING.END_TAG, searchStart);
        if (pos === -1) return -1;
        const after = buffer.slice(pos + KIRO_THINKING.END_TAG.length);
        if (after.startsWith('\n\n')) return pos;
        searchStart = pos + 1;
    }
}

/**
 * Find a "real" thinking end tag only when it is at the buffer end (after it is whitespace only).
 * This is used for boundary-event scenarios (tool_use starts immediately after thinking, or stream end).
 */
function findRealThinkingEndTagAtBufferEnd(buffer, startIndex = 0) {
    let searchStart = Math.max(0, startIndex);
    while (true) {
        const pos = findRealTag(buffer, KIRO_THINKING.END_TAG, searchStart);
        if (pos === -1) return -1;
        const after = buffer.slice(pos + KIRO_THINKING.END_TAG.length);
        if (isWhitespaceOnly(after)) return pos;
        searchStart = pos + 1;
    }
}

function findRealThinkingEndTagBeforeText(buffer, startIndex = 0) {
    return findRealTag(buffer, KIRO_THINKING.END_TAG, Math.max(0, startIndex));
}

/**
 * 通用的括号匹配函数 - 支持多种括号类型
 * @param {string} text - 要搜索的文本
 * @param {number} startPos - 起始位置
 * @param {string} openChar - 开括号字符 (默认 '[')
 * @param {string} closeChar - 闭括号字符 (默认 ']')
 * @returns {number} 匹配的闭括号位置，未找到返回 -1
 */
function findMatchingBracket(text, startPos, openChar = '[', closeChar = ']') {
    if (!text || startPos >= text.length || text[startPos] !== openChar) {
        return -1;
    }

    let bracketCount = 1;
    let inString = false;
    let escapeNext = false;

    for (let i = startPos + 1; i < text.length; i++) {
        const char = text[i];

        if (escapeNext) {
            escapeNext = false;
            continue;
        }

        if (char === '\\' && inString) {
            escapeNext = true;
            continue;
        }

        if (char === '"' && !escapeNext) {
            inString = !inString;
            continue;
        }

        if (!inString) {
            if (char === openChar) {
                bracketCount++;
            } else if (char === closeChar) {
                bracketCount--;
                if (bracketCount === 0) {
                    return i;
                }
            }
        }
    }
    return -1;
}


/**
 * 尝试修复常见的 JSON 格式问题
 * @param {string} jsonStr - 可能有问题的 JSON 字符串
 * @returns {string} 修复后的 JSON 字符串
 */
function repairJson(jsonStr) {
    let repaired = jsonStr;
    // 移除尾部逗号
    repaired = repaired.replace(/,\s*([}\]])/g, '$1');
    // 为未引用的键添加引号
    repaired = repaired.replace(/([{,]\s*)([a-zA-Z0-9_]+?)\s*:/g, '$1"$2":');
    // 确保字符串值被正确引用
    repaired = repaired.replace(/:\s*([a-zA-Z0-9_]+)(?=[,\}\]])/g, ':"$1"');
    return repaired;
}

function createToolCallTruncatedError(toolCall, reason) {
    return {
        type: 'error',
        error: {
            type: 'tool_call_truncated',
            message: `The upstream service returned an incomplete tool call for '${toolCall?.name || 'unknown_tool'}'.`,
            tool_use_id: toolCall?.toolUseId,
            tool_name: toolCall?.name,
            reason
        }
    };
}

function normalizeResponseFormat(format) {
    if (!format || typeof format !== 'object') return format;
    if (format.type !== 'json_schema') return format;
    if (format.json_schema) return format;
    if (!format.schema) return format;

    const { type, schema, name, description, strict, ...rest } = format;
    return {
        type,
        json_schema: {
            name: name || 'response',
            description,
            schema,
            strict,
            ...rest
        }
    };
}

function sanitizeProviderLeakText(text) {
    if (typeof text !== 'string' || !text) return text;
    return text
        .replace(/\bKiroIDE(?:-[A-Za-z0-9._-]+)*\b/g, 'Claude')
        .replace(/\bKiro\s+(API|service|provider|gateway|client|IDE|backend|upstream|transport|routing layer)\b/gi, 'Claude $1');
}

function sanitizeProviderLeakInObject(value) {
    if (typeof value === 'string') return sanitizeProviderLeakText(value);
    if (Array.isArray(value)) return value.map(item => sanitizeProviderLeakInObject(item));
    if (!value || typeof value !== 'object') return value;

    const copy = { ...value };
    for (const [key, child] of Object.entries(copy)) {
        copy[key] = sanitizeProviderLeakInObject(child);
    }
    return copy;
}

function rewriteClaudeCodeIdentityText(text) {
    if (typeof text !== 'string' || !text) return text;
    return text
        .replace(/You are Claude Code\b/g, 'You are Claude')
        .replace(/\bClaude Code\b/g, 'Claude')
        .replace(/\bAnthropic's official CLI\b/g, 'an official assistant interface');
}

function rewriteClaudeCodeIdentityInSystemEntry(entry) {
    if (typeof entry === 'string') {
        return rewriteClaudeCodeIdentityText(entry);
    }
    if (!entry || typeof entry !== 'object') {
        return entry;
    }
    if (typeof entry.text !== 'string') {
        return entry;
    }
    return {
        ...entry,
        text: rewriteClaudeCodeIdentityText(entry.text)
    };
}

const DEFAULT_VERIFIED_RECENT_FACTS = [
    'As of 2026, Donald J. Trump is the 45th and 47th President of the United States, and JD Vance is the Vice President of the United States.',
    'The Milano Cortina 2026 Olympic Winter Games were scheduled for February 6-22, 2026.',
    'The FIFA World Cup 2026 is scheduled for June 11-July 19, 2026, and is hosted across Canada, Mexico, and the United States.',
    'The FIFA World Cup 2026 is the first men\'s World Cup planned with 48 teams.'
];

function normalizeRecentFacts(facts) {
    return Array.isArray(facts)
        ? facts
            .filter(fact => typeof fact === 'string')
            .map(fact => fact.trim())
            .filter(Boolean)
        : [];
}

function buildRecentKnowledgeHint(content, config = {}) {
    const text = String(content || '');
    if (!text) return '';

    if (config?.KIRO_ENABLE_RECENT_KNOWLEDGE_HINT === false) return '';

    const configuredFacts = normalizeRecentFacts(config?.VERIFIED_RECENT_FACTS);
    const facts = config?.KIRO_USE_DEFAULT_RECENT_FACTS === false
        ? configuredFacts
        : [...DEFAULT_VERIFIED_RECENT_FACTS, ...configuredFacts];
    const uniqueFacts = [...new Set(facts)];
    if (uniqueFacts.length === 0) return '';
    return `<verified_recent_facts>\n${uniqueFacts.map(fact => `- ${fact}`).join('\n')}\n</verified_recent_facts>`;
}

/**
 * 从损坏的 JSON 中提取关键凭证字段
 * 当标准 JSON 解析和 repairJson 都失败时使用
 * @param {string} content - 文件内容
 * @returns {Object|null} 提取的凭证对象或 null
 */
function extractCredentialsFromCorruptedJson(content) {
    const extracted = {};

    // 定义需要提取的关键字段及其正则模式
    const fieldPatterns = {
        refreshToken: /"refreshToken"\s*:\s*"([^"]+)"/,
        accessToken: /"accessToken"\s*:\s*"([^"]+)"/,
        clientId: /"clientId"\s*:\s*"([^"]+)"/,
        clientSecret: /"clientSecret"\s*:\s*"([^"]+)"/,
        profileArn: /"profileArn"\s*:\s*"([^"]+)"/,
        region: /"region"\s*:\s*"([^"]+)"/,
        authMethod: /"authMethod"\s*:\s*"([^"]+)"/,
        expiresAt: /"expiresAt"\s*:\s*"([^"]+)"/,
        startUrl: /"startUrl"\s*:\s*"([^"]+)"/,
    };

    for (const [field, pattern] of Object.entries(fieldPatterns)) {
        const match = content.match(pattern);
        if (match && match[1]) {
            extracted[field] = match[1];
        }
    }

    // 至少需要 refreshToken 或 accessToken 才算有效
    if (extracted.refreshToken || extracted.accessToken) {
        logger.info(`[Kiro Auth] Extracted ${Object.keys(extracted).length} fields from corrupted JSON: ${Object.keys(extracted).join(', ')}`);
        return extracted;
    }

    return null;
}

/**
 * 解析单个工具调用文本
 * @param {string} toolCallText - 工具调用文本
 * @returns {Object|null} 解析后的工具调用对象或 null
 */
function parseSingleToolCall(toolCallText) {
    const namePattern = /\[Called\s+([^\s\]]+)\s+with\s+args:/i;
    const nameMatch = toolCallText.match(namePattern);

    if (!nameMatch) {
        return null;
    }

    const functionName = nameMatch[1].trim();
    const argsStartMarker = "with args:";
    const argsStartPos = toolCallText.toLowerCase().indexOf(argsStartMarker.toLowerCase());

    if (argsStartPos === -1) {
        return null;
    }

    const argsStart = argsStartPos + argsStartMarker.length;
    const argsEnd = toolCallText.lastIndexOf(']');

    if (argsEnd <= argsStart) {
        return null;
    }

    const jsonCandidate = toolCallText.substring(argsStart, argsEnd).trim();

    try {
        const repairedJson = repairJson(jsonCandidate);
        const argumentsObj = JSON.parse(repairedJson);

        if (typeof argumentsObj !== 'object' || argumentsObj === null) {
            return null;
        }

        const toolCallId = `call_${uuidv4().replace(/-/g, '').substring(0, 8)}`;
        return {
            id: toolCallId,
            type: "function",
            function: {
                name: functionName,
                arguments: JSON.stringify(argumentsObj)
            }
        };
    } catch (e) {
        logger.error(`Failed to parse tool call arguments: ${e.message}`, jsonCandidate);
        return null;
    }
}

function parseBracketToolCalls(responseText) {
    if (!responseText || !responseText.includes("[Called")) {
        return null;
    }

    const toolCalls = [];
    const callPositions = [];
    let start = 0;
    while (true) {
        const pos = responseText.indexOf("[Called", start);
        if (pos === -1) {
            break;
        }
        callPositions.push(pos);
        start = pos + 1;
    }

    for (let i = 0; i < callPositions.length; i++) {
        const startPos = callPositions[i];
        let endSearchLimit;
        if (i + 1 < callPositions.length) {
            endSearchLimit = callPositions[i + 1];
        } else {
            endSearchLimit = responseText.length;
        }

        const segment = responseText.substring(startPos, endSearchLimit);
        const bracketEnd = findMatchingBracket(segment, 0);

        let toolCallText;
        if (bracketEnd !== -1) {
            toolCallText = segment.substring(0, bracketEnd + 1);
        } else {
            // Fallback: if no matching bracket, try to find the last ']' in the segment
            const lastBracket = segment.lastIndexOf(']');
            if (lastBracket !== -1) {
                toolCallText = segment.substring(0, lastBracket + 1);
            } else {
                continue; // Skip this one if no closing bracket found
            }
        }
        
        const parsedCall = parseSingleToolCall(toolCallText);
        if (parsedCall) {
            toolCalls.push(parsedCall);
        }
    }
    return toolCalls.length > 0 ? toolCalls : null;
}

function deduplicateToolCalls(toolCalls) {
    const seen = new Set();
    const uniqueToolCalls = [];

    for (const tc of toolCalls) {
        const name = tc.function?.name || tc.name || 'unknown_tool';
        const args = tc.function?.arguments !== undefined ? tc.function.arguments : stringifyInput(tc.input || {});
        const key = `${name}-${args}`;
        if (!seen.has(key)) {
            seen.add(key);
            uniqueToolCalls.push(tc);
        } else {
            logger.info(`Skipping duplicate tool call: ${name}`);
        }
    }
    return uniqueToolCalls;
}

function normalizeToolCallArguments(args) {
    if (args === undefined || args === null) {
        return {};
    }

    let inputObject;
    try {
        inputObject = typeof args === 'string' ? JSON.parse(args) : args;
    } catch (e) {
        logger.warn(`[Kiro] Invalid JSON for tool call arguments. Wrapping in raw_arguments. Error: ${e.message}`, args);
        inputObject = { raw_arguments: args };
    }

    if (inputObject && typeof inputObject === 'object' && !Array.isArray(inputObject)) {
        const keys = Object.keys(inputObject);
        if (keys.length === 1 && Object.prototype.hasOwnProperty.call(inputObject, 'raw_arguments')) {
            const rawArguments = inputObject.raw_arguments;
            if (typeof rawArguments === 'string') {
                try {
                    const parsedRaw = JSON.parse(rawArguments);
                    if (parsedRaw && typeof parsedRaw === 'object' && !Array.isArray(parsedRaw)) {
                        return parsedRaw;
                    }
                } catch {
                    return { raw_arguments: rawArguments };
                }
            } else if (rawArguments && typeof rawArguments === 'object' && !Array.isArray(rawArguments)) {
                return rawArguments;
            }
        }
    }

    return inputObject && typeof inputObject === 'object' && !Array.isArray(inputObject)
        ? inputObject
        : { raw_arguments: inputObject };
}

function getKiroRequestUrl(model, baseUrl) {
    if (String(model || '').startsWith('amazonq')) {
        logger.warn('[Kiro] amazonq model requested but no Amazon Q endpoint is configured; falling back to Kiro baseUrl.');
    }
    return baseUrl;
}

function firstNonEmptyString(...values) {
    for (const value of values) {
        if (typeof value === 'string' && value.trim()) {
            return value.trim();
        }
    }
    return null;
}

function isToolHeavyRequest(messages, tools) {
    if (!Array.isArray(tools) || tools.length === 0) return false;
    const toolNames = tools
        .map(tool => String(tool?.name || '').toLowerCase())
        .join(' ');
    if (/(read|grep|glob|list|ls|cat|search|find|bash|shell)/.test(toolNames)) {
        return true;
    }

    const recentText = Array.isArray(messages)
        ? messages.slice(-3).map(message => {
            if (typeof message?.content === 'string') return message.content;
            if (Array.isArray(message?.content)) {
                return message.content.map(part => part?.text || part?.content || '').join(' ');
            }
            return '';
        }).join(' ').toLowerCase()
        : '';
    return /(read file|读取|grep|search|查找|list files|列出|run command|执行命令)/.test(recentText);
}

export class KiroApiService {
    constructor(config = {}) {
        this.isInitialized = false;
        this.config = config;
        this.credPath = config.KIRO_OAUTH_CREDS_DIR_PATH || path.join(os.homedir(), ".aws", "sso", "cache");
        this.credsBase64 = config.KIRO_OAUTH_CREDS_BASE64;
        this.useSystemProxy = config?.USE_SYSTEM_PROXY_KIRO ?? false;
        this.uuid = config?.uuid; // 获取多节点配置的 uuid
        logger.info(`[Kiro] System proxy ${this.useSystemProxy ? 'enabled' : 'disabled'}`);
        // Add kiro-oauth-creds-base64 and kiro-oauth-creds-file to config
        if (config.KIRO_OAUTH_CREDS_BASE64) {
            try {
                const decodedCreds = Buffer.from(config.KIRO_OAUTH_CREDS_BASE64, 'base64').toString('utf8');
                const parsedCreds = JSON.parse(decodedCreds);
                // Store parsedCreds to be merged in initializeAuth
                this.base64Creds = parsedCreds;
                logger.info('[Kiro] Successfully decoded Base64 credentials in constructor.');
            } catch (error) {
                logger.error(`[Kiro] Failed to parse Base64 credentials in constructor: ${error.message}`);
            }
        } else if (config.KIRO_OAUTH_CREDS_FILE_PATH) {
            this.credsFilePath = config.KIRO_OAUTH_CREDS_FILE_PATH;
        }

        this.modelName = KIRO_CONSTANTS.DEFAULT_MODEL_NAME;
        this.axiosInstance = null; // Initialize later in async method
        this.axiosSocialRefreshInstance = null;
    }
 
    async initialize() {
        if (this.isInitialized) return;
        logger.info('[Kiro] Initializing Kiro API Service...');
        // 注意：V2 读写分离架构下，初始化不再执行同步认证/刷新逻辑
        // 仅执行基础的凭证加载
        await this.loadCredentials();
        
        // 根据当前加载的凭证生成唯一的 Machine ID
        const machineId = generateMachineIdFromConfig({
            uuid: this.uuid,
            profileArn: this.profileArn,
            clientId: this.clientId
        });
        const kiroVersion = this.config.KIRO_VERSION || KIRO_CONSTANTS.KIRO_VERSION;
        const { osName, nodeVersion } = getSystemRuntimeInfo(this.config);

        // 配置 HTTP/HTTPS agent：开启 keep-alive，保留更多空闲连接以复用 TLS 握手。
        // maxFreeSockets 默认 5 太低，会频繁释放热连接导致下次请求重新握手（多几百 ms）。
        // scheduling: 'lifo' 优先复用最近使用的 socket，命中 TCP/TLS 缓存更好。
        const keepAliveMsecs = 30_000;
        const httpAgent = new http.Agent({
            keepAlive: true,
            keepAliveMsecs,
            maxSockets: 100,
            maxFreeSockets: 50,
            scheduling: 'lifo',
            timeout: KIRO_CONSTANTS.AXIOS_TIMEOUT,
        });
        const httpsAgent = new https.Agent({
            keepAlive: true,
            keepAliveMsecs,
            maxSockets: 100,
            maxFreeSockets: 50,
            scheduling: 'lifo',
            timeout: KIRO_CONSTANTS.AXIOS_TIMEOUT,
        });
        
        const isTLSSidecarEnabled = isTLSSidecarEnabledForProvider(this.config, this.config.MODEL_PROVIDER || MODEL_PROVIDER.KIRO_API);
        
        const axiosConfig = {
            timeout: KIRO_CONSTANTS.AXIOS_TIMEOUT,
            headers: {
                'Content-Type': KIRO_CONSTANTS.CONTENT_TYPE_JSON,
                'Accept': KIRO_CONSTANTS.ACCEPT_JSON,
                // amz-sdk-invocation-id: generated fresh per-request in callApi/streamApiReal
                // amz-sdk-request: generated per-request with correct attempt counter
                'x-amzn-codewhisperer-optout': true,
                // x-amzn-kiro-agent-mode: set per-request based on content (vibe/code)
                'x-amz-user-agent': `aws-sdk-js/1.0.34 KiroIDE-${kiroVersion}-${machineId}`,
                'user-agent': `aws-sdk-js/1.0.34 ua/2.1 os/${osName} lang/js md/electron#${nodeVersion} api/codewhispererstreaming#1.0.34 m/E KiroIDE-${kiroVersion}-${machineId}`,
            },
        };

        // 如果启用了 TLS Sidecar，就不配置 httpAgent 和 httpsAgent，避免配置冲突
        if (!isTLSSidecarEnabled) {
            axiosConfig.httpAgent = httpAgent;
            axiosConfig.httpsAgent = httpsAgent;
            // 配置自定义代理
            configureAxiosProxy(axiosConfig, this.config, this.config.MODEL_PROVIDER || MODEL_PROVIDER.KIRO_API);
        }
        
        this.axiosInstance = axios.create(axiosConfig);

        // Social refresh 使用独立的 axios 实例，仅设置最小必须 headers
        // 避免主实例 headers 中的 Kiro-specific 字段干扰 refreshToken 端点
        const socialRefreshAxiosConfig = {
            timeout: KIRO_CONSTANTS.TOKEN_REFRESH_TIMEOUT,
            headers: {
                'Content-Type': KIRO_CONSTANTS.CONTENT_TYPE_JSON,
            },
        };
        // 如果未启用 TLS Sidecar，复用主实例的代理配置
        if (!isTLSSidecarEnabled) {
            socialRefreshAxiosConfig.httpAgent = httpAgent;
            socialRefreshAxiosConfig.httpsAgent = httpsAgent;
            configureAxiosProxy(socialRefreshAxiosConfig, this.config, this.config.MODEL_PROVIDER || MODEL_PROVIDER.KIRO_API);
        }
        this.axiosSocialRefreshInstance = axios.create(socialRefreshAxiosConfig);
        this.isInitialized = true;
    }

    _applySidecar(axiosConfig) {
        return configureTLSSidecar(axiosConfig, this.config, this.config.MODEL_PROVIDER || MODEL_PROVIDER.KIRO_API);
    }

/**
 * 加载凭证信息（不执行刷新）
 */
async loadCredentials() {
    // 获取凭证文件路径
    const tokenFilePath = this.credsFilePath || path.join(this.credPath, KIRO_AUTH_TOKEN_FILE);

    // Helper to load credentials from a file
    const loadCredentialsFromFile = async (filePath) => {
        try {
            const fileContent = await fs.readFile(filePath, 'utf8');
            try {
                return JSON.parse(fileContent);
            } catch (parseError) {
                logger.warn('[Kiro Auth] JSON parse failed, attempting repair...');
                try {
                    const repaired = repairJson(fileContent);
                    const result = JSON.parse(repaired);
                    logger.info('[Kiro Auth] JSON repair successful');
                    return result;
                } catch (repairError) {
                    logger.warn('[Kiro Auth] JSON repair failed, attempting field extraction...');
                    // 尝试从损坏的 JSON 中提取关键字段
                    const extracted = extractCredentialsFromCorruptedJson(fileContent);
                    if (extracted) {
                        logger.info('[Kiro Auth] Field extraction successful, credentials recovered');
                        return extracted;
                    }
                    logger.error('[Kiro Auth] All recovery methods failed:', repairError.message);
                    return null;
                }
            }
        } catch (error) {
            if (error.code === 'ENOENT') {
                logger.debug(`[Kiro Auth] Credential file not found: ${filePath}`);
            } else {
                logger.warn(`[Kiro Auth] Failed to read credential file ${filePath}: ${error.message}`);
            }
            return null;
        }
    };

    try {
        let mergedCredentials = {};

        // Priority 1: Load from Base64 credentials if available
        if (this.base64Creds) {
            Object.assign(mergedCredentials, this.base64Creds);
            logger.info('[Kiro Auth] Successfully loaded credentials from Base64 (constructor).');
            this.base64Creds = null;
        }

        // 从文件加载
        const targetFilePath = tokenFilePath;
        const dirPath = path.dirname(targetFilePath);
        const targetFileName = path.basename(targetFilePath);

        logger.debug(`[Kiro Auth] Loading credentials from directory: ${dirPath}`);

        try {
            const targetCredentials = await loadCredentialsFromFile(targetFilePath);
            if (targetCredentials) {
                Object.assign(mergedCredentials, targetCredentials);
                logger.info(`[Kiro Auth] Successfully loaded OAuth credentials from ${targetFilePath}`);
            }

            const files = (await fs.readdir(dirPath)).sort((a, b) => a.localeCompare(b));
            for (const file of files) {
                if (file.endsWith('.json') && file !== targetFileName) {
                    const filePath = path.join(dirPath, file);
                    const credentials = await loadCredentialsFromFile(filePath);
                    if (credentials) {
                        for (const field of ['clientId', 'clientSecret', 'region', 'idcRegion', 'profileArn', 'startUrl']) {
                            if (mergedCredentials[field] === undefined && credentials[field] !== undefined) {
                                mergedCredentials[field] = credentials[field];
                            }
                        }
                        logger.debug(`[Kiro Auth] Loaded supplemental client credentials from ${file}`);
                    }
                }
            }
        } catch (error) {
            logger.warn(`[Kiro Auth] Error loading credentials from directory ${dirPath}: ${error.message}`);
        }

        // Apply loaded credentials. Force-refresh paths must not keep stale in-memory tokens.
        const applyCredential = (field) => {
            if (mergedCredentials[field] !== undefined && mergedCredentials[field] !== null) {
                this[field] = mergedCredentials[field];
            }
        };
        applyCredential('accessToken');
        applyCredential('refreshToken');
        applyCredential('clientId');
        applyCredential('clientSecret');
        applyCredential('authMethod');
        applyCredential('expiresAt');
        applyCredential('profileArn');
        applyCredential('region');
        applyCredential('authRegion');
        applyCredential('apiRegion');
        applyCredential('idcRegion');

        const defaultRegion = firstNonEmptyString(this.config.KIRO_REGION, KIRO_CONSTANTS.DEFAULT_REGION);
        // Keep auth and API regions independent, matching kiro.rs:
        // credentials.region is a backward-compatible auth-region field.
        this.authRegion = firstNonEmptyString(
            this.authRegion,
            this.idcRegion,
            this.region,
            this.config.KIRO_AUTH_REGION,
            defaultRegion
        );
        this.apiRegion = firstNonEmptyString(
            this.apiRegion,
            this.config.KIRO_API_REGION,
            this.config.KIRO_REGION,
            defaultRegion
        );
        this.region = firstNonEmptyString(this.region, defaultRegion);
        this.idcRegion = firstNonEmptyString(this.idcRegion, this.authRegion);

        // idcRegion 用于 REFRESH_IDC_URL，如果未设置则使用 region
        if (!this.idcRegion) {
            this.idcRegion = this.authRegion;
        }

        this.refreshUrl = (this.config.KIRO_REFRESH_URL || KIRO_CONSTANTS.REFRESH_URL).replace("{{region}}", this.authRegion);
        this.refreshIDCUrl = (this.config.KIRO_REFRESH_IDC_URL || KIRO_CONSTANTS.REFRESH_IDC_URL).replace("{{region}}", this.idcRegion);
        this.baseUrl = (this.config.KIRO_BASE_URL || KIRO_CONSTANTS.BASE_URL).replace("{{region}}", this.apiRegion);
        logger.info(`[Kiro Auth] Region resolved: authRegion=${this.authRegion}, apiRegion=${this.apiRegion}`);
    } catch (error) {
        logger.warn(`[Kiro Auth] Error during credential loading: ${error.message}`);
    }
}

async initializeAuth(forceRefresh = false) {
    const hasValidAccessToken = this.accessToken && !this.isTokenExpired();
    if (hasValidAccessToken && !forceRefresh) {
        logger.debug('[Kiro Auth] Access token already available and not forced refresh.');
        return;
    }

    // 首先执行基础凭证加载
    await this.loadCredentials();

    // 只有在明确要求强制刷新，或者 AccessToken 确实缺失时，才执行刷新
    // 注意：在 V2 架构下，此方法主要由 PoolManager 的后台队列调用
    if (forceRefresh || (!hasValidAccessToken && this.refreshToken)) {
        if (!this.refreshToken) {
            throw new Error('No refresh token available to refresh access token.');
        }

        const tokenFilePath = this.credsFilePath || path.join(this.credPath, KIRO_AUTH_TOKEN_FILE);
        await this._doTokenRefresh(this.saveCredentialsToFile.bind(this), tokenFilePath);
    }

    if (!this.accessToken) {
        throw new Error('No access token available after initialization and refresh attempts.');
    }
}

/**
 * Helper to save credentials
 */
async saveCredentialsToFile(filePath, newData) {
    await withFileLock(filePath, async () => {
        let existingData = {};
        try {
            const fileContent = await fs.readFile(filePath, 'utf8');
            try {
                existingData = JSON.parse(fileContent);
            } catch (parseError) {
                logger.warn('[Kiro Auth] JSON parse failed, attempting repair...');
                try {
                    const repaired = repairJson(fileContent);
                    existingData = JSON.parse(repaired);
                    logger.info('[Kiro Auth] JSON repair successful');
                } catch (repairError) {
                    logger.warn('[Kiro Auth] JSON repair failed, attempting field extraction...');
                    const extracted = extractCredentialsFromCorruptedJson(fileContent);
                    if (extracted) {
                        existingData = extracted;
                        logger.info('[Kiro Auth] Field extraction successful');
                    } else {
                        logger.error('[Kiro Auth] All recovery methods failed:', repairError.message);
                        existingData = {};
                    }
                }
            }
        } catch (readError) {
            if (readError.code === 'ENOENT') {
                logger.debug(`[Kiro Auth] Token file not found, creating new one: ${filePath}`);
            } else {
                logger.warn(`[Kiro Auth] Could not read existing token file ${filePath}: ${readError.message}`);
            }
        }
        const mergedData = { ...existingData, ...newData };
        await atomicWriteFile(filePath, JSON.stringify(mergedData, null, 2), { encoding: 'utf8', mode: 0o600 });
    });
    logger.info(`[Kiro Auth] Updated token file: ${filePath}`);
};

    /**
     * 执行实际的 token 刷新操作（内部方法）
     * @param {Function} saveCredentialsToFile - 保存凭证的函数
     * @param {string} tokenFilePath - 凭证文件路径
     */
    async _doTokenRefresh(saveCredentialsToFile, tokenFilePath) {
        try {
            const requestBody = {
                refreshToken: this.refreshToken,
            };

            const hasIdcClientCredentials = !!(this.clientId && this.clientSecret);
            const isSocialAuth = this.authMethod === KIRO_CONSTANTS.AUTH_METHOD_SOCIAL ||
                (!this.authMethod && !hasIdcClientCredentials);
            if (!this.authMethod) {
                this.authMethod = isSocialAuth ? KIRO_CONSTANTS.AUTH_METHOD_SOCIAL : 'builder-id';
                logger.warn(`[Kiro Auth] authMethod missing in credentials. Inferred ${this.authMethod} from available fields.`);
            }

            let refreshUrl = this.refreshUrl;
            if (!isSocialAuth) {
                refreshUrl = this.refreshIDCUrl;
                if (!hasIdcClientCredentials) {
                    throw new Error('IDC refresh requires clientId and clientSecret.');
                }
                requestBody.clientId = this.clientId;
                requestBody.clientSecret = this.clientSecret;
                requestBody.grantType = 'refresh_token';
            }

            let response = null;
            // 使用更短的超时时间进行 token 刷新，避免阻塞其他请求
            const refreshConfig = { timeout: KIRO_CONSTANTS.TOKEN_REFRESH_TIMEOUT };
            
            const axiosConfig = {
                method: 'post',
                url: refreshUrl,
                data: requestBody,
                ...refreshConfig
            };
            this._applySidecar(axiosConfig);

            if (isSocialAuth) {
                response = await this.axiosSocialRefreshInstance.request(axiosConfig);
                logger.info('[Kiro Auth] Token refresh social response: ok');
            } else {
                // Builder ID / IDC refresh must NOT use the main axiosInstance
                // because it carries Kiro-specific headers (x-amzn-kiro-agent-mode,
                // x-amzn-codewhisperer-optout, etc.) that shouldn't go to the OIDC
                // endpoint and could be used for fingerprinting.
                axiosConfig.headers = {
                    'Content-Type': 'application/json',
                    'User-Agent': 'KiroIDE',
                };
                response = await this.axiosSocialRefreshInstance.request(axiosConfig);
                logger.info('[Kiro Auth] Token refresh idc response: ok');
            }

            if (response.data && response.data.accessToken) {
                this.accessToken = response.data.accessToken;
                this.refreshToken = response.data.refreshToken || this.refreshToken;
                this.profileArn = response.data.profileArn || this.profileArn;
                const expiresIn = Number(response.data.expiresIn) || 3600;
                const expiresAt = new Date(Date.now() + expiresIn * 1000).toISOString();
                this.expiresAt = expiresAt;
                logger.info('[Kiro Auth] Access token refreshed successfully');

                const updatedTokenData = {
                    accessToken: this.accessToken,
                    refreshToken: this.refreshToken,
                    expiresAt: expiresAt,
                };
                if (this.profileArn) {
                    updatedTokenData.profileArn = this.profileArn;
                }
                await saveCredentialsToFile(tokenFilePath, updatedTokenData);

                // 刷新成功，重置 PoolManager 中的刷新状态并标记为健康
                const poolManager = getProviderPoolManager();
                if (poolManager && this.uuid) {
                    poolManager.resetProviderRefreshStatus(this.config.MODEL_PROVIDER || MODEL_PROVIDER.KIRO_API, this.uuid);
                }
            } else {
                throw new Error('Invalid refresh response: Missing accessToken');
            }
        } catch (error) {
            logger.error('[Kiro Auth] Token refresh failed:', error.message);
            throw new Error(`Token refresh failed: ${error.message}`);
        }
    }


    /**
     * Count tokens for a given text using Claude's official tokenizer
     * Static version for use without instance
     */
    static countTextTokens(text) {
        return countTextTokensUtil(text);
    }

    /**
     * Count tokens for a message request (compatible with Anthropic API)
     * Static version for use without instance
     */
    static countTokens(requestBody) {
        return countTokensUtil(requestBody);
    }

    /**
     * Calculate input tokens from request body
     * Static version for use without instance
     */
    static estimateInputTokens(requestBody) {
        return estimateInputTokensUtil(requestBody);
    }

    /**
     * Extract text content from OpenAI message format
     */
    getContentText(message) {
        return getContentTextUtil(message);
    }

    /**
     * 清洗 tool_use 的 input 对象，移除空字符串 key 等不合法字段
     * Kiro API 不接受空字符串 key 的 JSON 对象（如 {"": "value"}）
     */
    _sanitizeToolInput(input) {
        if (!input || typeof input !== 'object' || Array.isArray(input)) {
            return input;
        }
        const sanitized = {};
        for (const [key, value] of Object.entries(input)) {
            if (key === '') {
                logger.info(`[Kiro] Removed empty-string key from tool input, value: ${String(value).substring(0, 100)}`);
                continue;
            }
            sanitized[key] = value;
        }
        return sanitized;
    }

    _isEmptyToolInput(input) {
        if (!input || typeof input !== 'object' || Array.isArray(input)) {
            return false;
        }
        return Object.keys(this._sanitizeToolInput(input)).length === 0;
    }

    _getToolInputValidationError(toolName, input, tools = []) {
        const name = String(toolName || '').trim();
        const lowerName = name.toLowerCase();
        const normalizedInput = input && typeof input === 'object' && !Array.isArray(input)
            ? this._sanitizeToolInput(input)
            : {};

        const tool = Array.isArray(tools)
            ? tools.find(candidate => String(candidate?.name || '').toLowerCase() === lowerName)
            : null;
        const schema = tool?.input_schema || tool?.inputSchema || tool?.parameters || {};
        const required = Array.isArray(schema?.required) ? schema.required : [];
        const missing = required.filter(key => {
            const value = normalizedInput[key];
            return value === undefined || value === null || (typeof value === 'string' && value.trim() === '');
        });
        if (missing.length > 0) {
            return `missing required parameter(s): ${missing.join(', ')}`;
        }

        if (/(bash|shell)/i.test(name)) {
            const command = normalizedInput.command;
            if (typeof command !== 'string' || command.trim() === '') {
                return 'missing required parameter: command';
            }
        }

        const isFileOrShellTool = /read|edit|write|bash|shell|grep|glob|list|ls|search|find/i.test(name);
        if (isFileOrShellTool && Object.keys(normalizedInput).length === 0) {
            return 'empty input for file/shell tool';
        }

        return null;
    }

    _filterInvalidGeneratedToolCalls(toolCalls, tools = []) {
        if (!Array.isArray(toolCalls) || toolCalls.length === 0) return toolCalls;
        return toolCalls.filter(tc => {
            const name = tc.function?.name || tc.name || 'unknown_tool';
            const input = tc.function?.arguments !== undefined
                ? normalizeToolCallArguments(tc.function.arguments)
                : normalizeToolCallArguments(tc.input || {});
            const validationError = this._getToolInputValidationError(name, input, tools);
            if (validationError) {
                logger.warn(`[Kiro] Dropping invalid generated tool call '${name}': ${validationError}`);
                return false;
            }
            return true;
        });
    }

    _isToolResultError(part) {
        return part?.is_error === true || part?.status === 'error';
    }

    _buildKiroToolResult(part) {
        const resultText = this.getContentText(part?.content);
        const isError = this._isToolResultError(part);
        return {
            content: [{ text: isError ? `[Tool error]\n${resultText}` : resultText }],
            status: isError ? 'error' : 'success',
            toolUseId: part.tool_use_id
        };
    }

    _reconcileKiroToolHistory(history, currentToolResults = []) {
        const availableToolUseIds = new Set();
        const pairedToolUseIds = new Set();

        for (const item of history) {
            const assistant = item?.assistantResponseMessage;
            if (Array.isArray(assistant?.toolUses)) {
                for (const toolUse of assistant.toolUses) {
                    if (toolUse?.toolUseId) availableToolUseIds.add(toolUse.toolUseId);
                }
            }

            const user = item?.userInputMessage;
            const context = user?.userInputMessageContext;
            if (!Array.isArray(context?.toolResults)) continue;

            const filteredResults = [];
            for (const result of context.toolResults) {
                const id = result?.toolUseId;
                if (!id || !availableToolUseIds.has(id)) {
                    logger.warn(`[Kiro] Dropping orphaned historical tool_result: ${id || 'unknown'}`);
                    continue;
                }
                if (pairedToolUseIds.has(id)) {
                    logger.warn(`[Kiro] Dropping duplicate historical tool_result: ${id}`);
                    continue;
                }
                pairedToolUseIds.add(id);
                filteredResults.push(result);
            }

            if (filteredResults.length > 0) {
                context.toolResults = filteredResults;
            } else {
                delete context.toolResults;
            }
            if (Object.keys(context).length === 0) {
                delete user.userInputMessageContext;
            }
        }

        const filteredCurrentToolResults = [];
        for (const result of currentToolResults) {
            const id = result?.toolUseId;
            if (!id || !availableToolUseIds.has(id)) {
                logger.warn(`[Kiro] Dropping orphaned current tool_result: ${id || 'unknown'}`);
                continue;
            }
            if (pairedToolUseIds.has(id)) {
                logger.warn(`[Kiro] Dropping duplicate current tool_result: ${id}`);
                continue;
            }
            pairedToolUseIds.add(id);
            filteredCurrentToolResults.push(result);
        }

        const orphanedToolUseIds = new Set(
            [...availableToolUseIds].filter(id => !pairedToolUseIds.has(id))
        );
        if (orphanedToolUseIds.size > 0) {
            logger.warn(`[Kiro] Removing ${orphanedToolUseIds.size} orphaned historical tool_use(s) without matching tool_result`);
        }

        for (const item of history) {
            const assistant = item?.assistantResponseMessage;
            if (!Array.isArray(assistant?.toolUses)) continue;

            assistant.toolUses = assistant.toolUses.filter(toolUse => !orphanedToolUseIds.has(toolUse?.toolUseId));
            if (assistant.toolUses.length === 0) {
                delete assistant.toolUses;
            }
            if (!assistant.content || String(assistant.content).length === 0) {
                assistant.content = ' ';
            }
        }

        return filteredCurrentToolResults;
    }

    _formatToolUseAsText(part) {
        const name = part?.name || 'unknown_tool';
        const id = part?.id ? ` ${part.id}` : '';
        let inputText = '';
        try {
            inputText = JSON.stringify(part?.input ?? {}, null, 2);
        } catch (e) {
            inputText = String(part?.input ?? '');
        }
        return `\n\n[Tool call${id}: ${name}]\n${inputText}\n[/Tool call]\n`;
    }

    _formatToolResultAsText(part) {
        const id = part?.tool_use_id ? ` ${part.tool_use_id}` : '';
        const resultText = this.getContentText(part?.content);
        return `\n\n[Tool result${id}]\n${resultText}\n[/Tool result]\n`;
    }

    /**
     * 统一处理内容，将不同格式的内容转换为文本
     * @param {any} content - 内容对象或数组
     * @returns {string} 处理后的文本
     */
    processContent(content) {
        return processContentUtil(content);
    }

    _normalizeThinkingBudgetTokens(budgetTokens) {
        let value = Number(budgetTokens);
        if (!Number.isFinite(value) || value <= 0) {
            value = KIRO_THINKING.DEFAULT_BUDGET_TOKENS;
        }
        value = Math.floor(value);
        if (value < KIRO_THINKING.MIN_BUDGET_TOKENS) value = KIRO_THINKING.MIN_BUDGET_TOKENS;
        return Math.min(value, KIRO_THINKING.MAX_BUDGET_TOKENS);
    }

    _generateThinkingPrefix(thinking) {
        if (!thinking || typeof thinking !== 'object') return null;
        const type = String(thinking.type || '').toLowerCase().trim();
        if (type === 'disabled' || type === 'off') return null;

        if (type === 'enabled') {
            const budget = this._normalizeThinkingBudgetTokens(thinking.budget_tokens);
            return `<thinking_mode>enabled</thinking_mode><max_thinking_length>${budget}</max_thinking_length>`;
        }

        if (type === 'adaptive') {
            // CodeWhisperer may not support adaptive thinking mode natively.
            // Convert to 'enabled' with a budget derived from effort level to ensure compatibility.
            const effortRaw = typeof thinking.effort === 'string' ? thinking.effort : '';
            const effort = effortRaw.toLowerCase().trim();
            const effortBudgetMap = { 'low': 4096, 'medium': 12288, 'high': KIRO_THINKING.MAX_BUDGET_TOKENS };
            const budget = effortBudgetMap[effort] || KIRO_THINKING.MAX_BUDGET_TOKENS;
            logger.info(`[Kiro] Converting adaptive thinking (effort=${effort || 'default'}) to enabled mode with budget=${budget}`);
            return `<thinking_mode>enabled</thinking_mode><max_thinking_length>${budget}</max_thinking_length>`;
        }

        return null;
    }

    _getEffectiveThinking(messages, tools, thinking) {
        const type = String(thinking?.type || '').toLowerCase().trim();
        if ((type === 'enabled' || type === 'adaptive') && isToolHeavyRequest(messages, tools)) {
            logger.info('[Kiro] Disabling thinking for tool-heavy deterministic request.');
            return { type: 'disabled' };
        }
        return thinking;
    }

    _applyEffectiveThinkingToRequestBody(requestBody) {
        if (!requestBody?.thinking) return;
        const effectiveThinking = this._getEffectiveThinking(requestBody.messages, requestBody.tools, requestBody.thinking);
        if (effectiveThinking?.type === 'disabled') {
            delete requestBody.thinking;
        } else {
            requestBody.thinking = effectiveThinking;
        }
    }

    /**
     * Build a natural-language directive that emulates Anthropic's tool_choice parameter.
     * CodeWhisperer has no native tool_choice, so we inject instructions the model will obey.
     *
     * Supported shapes (matching Anthropic's API):
     *   - { type: "auto" }      → no instruction (default behavior)
     *   - { type: "any" }       → force some tool call
     *   - { type: "tool", name: "X" } → force specific tool
     *   - { type: "none" }      → forbid tool calls
     *   - "required" / "any"    → force some tool call (OpenAI-compat)
     *   - "none"                → forbid tool calls
     *   - "auto"                → no instruction
     */
    _buildToolChoiceInstruction(toolChoice, tools) {
        if (!toolChoice) return '';
        if (!Array.isArray(tools) || tools.length === 0) return '';

        let type = '';
        let forcedName = '';
        if (typeof toolChoice === 'string') {
            type = toolChoice.toLowerCase().trim();
        } else if (typeof toolChoice === 'object') {
            type = String(toolChoice.type || '').toLowerCase().trim();
            forcedName = typeof toolChoice.name === 'string' ? toolChoice.name : '';
        }

        if (type === 'auto' || type === '') return '';

        if (type === 'none') {
            return '[Tool Usage Directive]\nYou MUST NOT call any tool in this turn. Respond with text only.';
        }

        if (type === 'tool' && forcedName) {
            return `[Tool Usage Directive]\nYou MUST invoke the tool named "${forcedName}" to answer this request. Do not respond with plain text before the tool call. The tool call is mandatory.`;
        }

        if (type === 'any' || type === 'required') {
            const availableNames = tools.map(t => t && t.name).filter(Boolean).join(', ');
            return `[Tool Usage Directive]\nYou MUST invoke one of the available tools (${availableNames}) to answer this request. A tool call is mandatory; do not respond with plain text only.`;
        }

        return '';
    }

    _buildToolReliabilityInstruction(tools) {
        if (!Array.isArray(tools) || tools.length === 0) return '';
        const toolNames = tools.map(tool => tool?.name).filter(Boolean);
        const lowerNames = toolNames.map(name => name.toLowerCase());
        const hasFileOrShellTools = lowerNames.some(name =>
            /read|edit|write|bash|shell|grep|glob|list|ls|search|find/.test(name)
        );
        if (!hasFileOrShellTools) return '';

        return [
            '[Tool Reliability Directive]',
            'When using file or shell tools, use only paths, working directories, and command arguments that are present in the conversation or tool results.',
            'Do not invent usernames, home directories, drive letters, or repository paths. If a path is relative, keep it relative unless an absolute path was explicitly provided.',
            'For Bash/shell tools, always provide the required command parameter as a non-empty string.',
            'For Read/Edit/Write tools, always provide the exact required path parameter from known context. If the path is unknown, inspect/list first instead of guessing.'
        ].join('\n');
    }

    _hasThinkingPrefix(text) {
        if (!text) return false;
        return text.includes(KIRO_THINKING.MODE_TAG) || text.includes(KIRO_THINKING.MAX_LEN_TAG);
    }

    /**
     * Determine whether the current message should be wrapped with a code-context hint
     * to help bypass AWS CodeWhisperer's intent classifier that silently filters non-code queries.
     * @param {string} content - The current message content
     * @param {Array} toolResults - Current tool results
     * @param {Object} toolsContext - Tools context object
     * @returns {boolean} True if the message should be wrapped
     */
    _shouldApplyCodeContextWrapper(content, toolResults, toolsContext) {
        if (!content || typeof content !== 'string') return false;
        // Don't wrap if tools are present (it's already a coding context)
        if (toolResults && toolResults.length > 0) return false;
        if (toolsContext && toolsContext.tools && toolsContext.tools.length > 0) return false;
        // Don't wrap if content is long enough (likely already has sufficient context)
        if (content.length > 500) return false;
        // Don't wrap if content already has code-like patterns
        const codePatterns = /(`{1,3}|function\s|class\s|import\s|require\(|def\s|const\s|let\s|var\s|public\s|private\s|\.py|\.js|\.ts|\.java|\.cpp|\.go|\.rs|<\/?[a-z]+>|\{[\s\S]*\})/i;
        if (codePatterns.test(content)) return false;
        return true;
    }

    _shouldApplyLightKiroContext(content, toolResults, toolsContext, thinking, responseFormat) {
        if (!this.config.KIRO_ENABLE_LIGHT_CONTEXT) return false;
        if (!this._shouldApplyCodeContextWrapper(content, toolResults, toolsContext)) return false;
        const thinkingType = typeof thinking?.type === 'string' ? thinking.type.toLowerCase() : '';
        const hasStructuredOutput = !!responseFormat;
        return hasStructuredOutput || thinkingType === 'enabled' || thinkingType === 'adaptive';
    }

    _toClaudeContentBlocksFromKiroText(content) {
        const raw = content ?? '';
        if (!raw) return [];
        
        const startPos = findRealTag(raw, KIRO_THINKING.START_TAG);
        if (startPos === -1) {
            return [{ type: "text", text: raw }];
        }
        
        const before = raw.slice(0, startPos);
        let rest = raw.slice(startPos + KIRO_THINKING.START_TAG.length);

        // Strip a single leading newline after `<thinking>` for cleaner blocks.
        if (rest.startsWith('\r\n')) rest = rest.slice(2);
        else if (rest.startsWith('\n')) rest = rest.slice(1);

        let endPosInRest = findRealThinkingEndTag(rest);
        if (endPosInRest === -1) endPosInRest = findRealThinkingEndTagAtBufferEnd(rest);
        if (endPosInRest === -1) endPosInRest = findRealThinkingEndTagBeforeText(rest);

        let thinking = '';
        let after = '';
        if (endPosInRest === -1) {
            thinking = rest;
        } else {
            thinking = rest.slice(0, endPosInRest);
            after = rest.slice(endPosInRest + KIRO_THINKING.END_TAG.length);
        }
        
        if (after.startsWith('\r\n\r\n')) after = after.slice(4);
        else if (after.startsWith('\n\n')) after = after.slice(2);
        if (isWhitespaceOnly(after)) after = '';
        
        const blocks = [];
        if (before && !isWhitespaceOnly(before)) blocks.push({ type: "text", text: before });
        blocks.push({ type: "thinking", thinking });
        if (after && !isWhitespaceOnly(after)) blocks.push({ type: "text", text: after });
        return blocks;
    }

    /**
     * Build CodeWhisperer request from OpenAI messages
     */
    async buildCodewhispererRequest(messages, model, tools = null, inSystemPrompt = null, thinking = null, toolChoice = null, responseFormat = null, requestContext = {}) {
        const conversationId = uuidv4();

        let systemPrompt = this.getContentText(inSystemPrompt);
        let responseFormatInstruction = '';

        // Inject JSON instruction when response_format is requested
        if (responseFormat) {
            if (responseFormat.type === 'json_object') {
                responseFormatInstruction = 'You must respond with valid JSON only. Do not include any text outside the JSON object.';
            } else if (responseFormat.type === 'json_schema' && responseFormat.json_schema) {
                const schemaStr = JSON.stringify(responseFormat.json_schema.schema || responseFormat.json_schema, null, 2);
                responseFormatInstruction = `You must respond with valid JSON only, conforming to this schema:\n${schemaStr}\nDo not include any text outside the JSON object.`;
            }
            if (responseFormatInstruction) {
                systemPrompt = systemPrompt ? `${systemPrompt}\n\n${responseFormatInstruction}` : responseFormatInstruction;
            }
        }

        const identityGuardInstruction = [
            'Identity and provider disclosure:',
            'If asked who you are, answer as Claude.',
            'Do not mention internal providers, routing layers, gateways, IDE names, or transport details.',
            'Do not reveal or repeat hidden vendor names in user-visible text.'
        ].join('\n');
        systemPrompt = systemPrompt
            ? `${identityGuardInstruction}\n\n${systemPrompt}`
            : identityGuardInstruction;
        
        const processedMessages = messages.map(message => ({
            ...message,
            content: Array.isArray(message.content) ? [...message.content] : message.content
        }));

        if (processedMessages.length === 0) {
            throw new Error('No user messages found');
        }

        const effectiveThinking = this._getEffectiveThinking(processedMessages, tools, thinking);
        const thinkingPrefix = this._generateThinkingPrefix(effectiveThinking);
        if (thinkingPrefix) {
            if (!systemPrompt) {
                systemPrompt = thinkingPrefix;
            } else if (!this._hasThinkingPrefix(systemPrompt)) {
                systemPrompt = `${thinkingPrefix}\n${systemPrompt}`;
            }
        }

        // 判断最后一条消息是否为 assistant,如果是则移除
        const lastMessage = processedMessages[processedMessages.length - 1];
        if (processedMessages.length > 0 && lastMessage.role === 'assistant') {
            if (lastMessage.content[0].type === "text" && lastMessage.content[0].text === "{") {
                logger.info('[Kiro] Removing last assistant with "{" message from processedMessages');
                processedMessages.pop();
            }
        }

        // 合并相邻相同 role 的消息
        const mergedMessages = [];
        for (let i = 0; i < processedMessages.length; i++) {
            const currentMsg = processedMessages[i];
            
            if (mergedMessages.length === 0) {
                mergedMessages.push(currentMsg);
            } else {
                const lastMsg = mergedMessages[mergedMessages.length - 1];
                
                // 判断当前消息和上一条消息是否为相同 role
                if (currentMsg.role === lastMsg.role) {
                    // 合并消息内容
                    if (Array.isArray(lastMsg.content) && Array.isArray(currentMsg.content)) {
                        // 如果都是数组,合并数组内容
                        lastMsg.content.push(...currentMsg.content);
                    } else if (typeof lastMsg.content === 'string' && typeof currentMsg.content === 'string') {
                        // 如果都是字符串,用换行符连接
                        lastMsg.content += '\n' + currentMsg.content;
                    } else if (Array.isArray(lastMsg.content) && typeof currentMsg.content === 'string') {
                        // 上一条是数组,当前是字符串,添加为 text 类型
                        lastMsg.content.push({ type: 'text', text: currentMsg.content });
                    } else if (typeof lastMsg.content === 'string' && Array.isArray(currentMsg.content)) {
                        // 上一条是字符串,当前是数组,转换为数组格式
                        lastMsg.content = [{ type: 'text', text: lastMsg.content }, ...currentMsg.content];
                    }
                    // logger.info(`[Kiro] Merged adjacent ${currentMsg.role} messages`);
                } else {
                    mergedMessages.push(currentMsg);
                }
            }
        }
        
        // 用合并后的消息替换原消息数组
        processedMessages.length = 0;
        processedMessages.push(...mergedMessages);

        const codewhispererModel = resolveKiroModel(model);
        const toolNameMaps = buildKiroToolNameMaps(tools);
        const toolsContext = buildKiroToolsContext(tools, toolNameMaps);
        const hasToolDefinitions = Array.isArray(tools) && tools.length > 0;

        // tool_choice 指令：CodeWhisperer 没有原生 tool_choice 参数。
        // 注意：不要注入到 system prompt——system 指令会被中间的 history 稀释，
        // 模型对"最近看到的指令"服从度更高，所以把它拼到当前 user 消息末尾，
        // 确保模型在决定输出前最后读到的是这条强约束。
        const toolChoiceInstruction = this._buildToolChoiceInstruction(toolChoice, tools);
        const toolReliabilityInstruction = this._buildToolReliabilityInstruction(tools);

        const history = [];
        let startIndex = 0;
        const validToolUseIds = new Set();
        const downgradedToolUseIds = new Set();

        let prependSystemToCurrentMessage = false;

        // Handle system prompt
        if (systemPrompt) {
            // Keep single-turn requests as a single current message. Duplicating the same user
            // payload into history and currentMessage can make Kiro reject large agent prompts.
            if (processedMessages[0].role === 'user' && processedMessages.length === 1) {
                prependSystemToCurrentMessage = true;
            } else if (processedMessages[0].role === 'user') {
                let firstUserContent = this.getContentText(processedMessages[0]);
                history.push({
                    userInputMessage: {
                        content: `${systemPrompt}\n\n${firstUserContent}`,
                        modelId: codewhispererModel,
                        origin: KIRO_CONSTANTS.ORIGIN_AI_EDITOR,
                    }
                });
                startIndex = 1; // Start processing from the second message
            } else {
                // If the first message is not a user message, or if there's no initial user message,
                // add system prompt as a standalone user message.
                history.push({
                    userInputMessage: {
                        content: systemPrompt,
                        modelId: codewhispererModel,
                        origin: KIRO_CONSTANTS.ORIGIN_AI_EDITOR,
                    }
                });
            }
        }

        // 保留最近 5 条历史消息中的图片
        const keepImageThreshold = 5;        
        for (let i = startIndex; i < processedMessages.length - 1; i++) {
            const message = processedMessages[i];
            // 计算当前消息距离最后一条消息的位置（从后往前数）
            const distanceFromEnd = (processedMessages.length - 1) - i;
            // 如果距离末尾不超过 5 条，则保留图片
            const shouldKeepImages = distanceFromEnd <= keepImageThreshold;
            
            if (message.role === 'user') {
                let userInputMessage = {
                    content: '',
                    modelId: codewhispererModel,
                    origin: KIRO_CONSTANTS.ORIGIN_AI_EDITOR
                };
                let imageCount = 0;
                let toolResults = [];
                let images = [];
                
                if (Array.isArray(message.content)) {
                    for (const part of message.content) {
                        if (part.type === 'text') {
                            userInputMessage.content += part.text;
                        } else if (part.type === 'tool_result') {
                            if (hasToolDefinitions && validToolUseIds.has(part.tool_use_id) && !downgradedToolUseIds.has(part.tool_use_id)) {
                                toolResults.push(this._buildKiroToolResult(part));
                            } else {
                                userInputMessage.content += this._formatToolResultAsText(part);
                            }
                        } else if (part.type === 'document') {
                            // Handle PDF / text documents by parsing to inline text
                            const mediaType = part?.source?.media_type || '';
                            if (mediaType === 'application/pdf' && part?.source?.type === 'base64' && part?.source?.data) {
                                const pdfText = await parsePdfBase64ToText(part.source.data);
                                if (pdfText) {
                                    const label = part?.title ? `PDF Document: ${part.title}` : 'PDF Document';
                                    userInputMessage.content += `\n\n[${label}]\n${pdfText}\n[End of Document]\n\n`;
                                    logger.info(`[Kiro] Parsed PDF in history (${pdfText.length} chars)`);
                                } else {
                                    userInputMessage.content += `\n[PDF document could not be parsed]\n`;
                                }
                            } else if (part?.source?.type === 'text' && part?.source?.data) {
                                userInputMessage.content += `\n\n[Document]\n${part.source.data}\n[End of Document]\n\n`;
                            } else if (part?.source?.type === 'url' && part?.source?.url) {
                                const label = part?.title ? `Document: ${part.title}` : 'Document';
                                userInputMessage.content += `\n\n[${label}]\n[Source URL: ${part.source.url}]\n[End of Document]\n\n`;
                            }
                        } else if (part.type === 'image') {
                            if (shouldKeepImages) {
                                // 最近 5 条消息内的图片保留原始数据
                                images.push({
                                    format: part.source.media_type.split('/')[1],
                                    source: {
                                        bytes: part.source.data
                                    }
                                });
                            } else {
                                // 超过 5 条历史记录的图片只记录数量
                                imageCount++;
                            }
                        }
                    }
                } else {
                    userInputMessage.content = this.getContentText(message);
                }
                
                // 如果有保留的图片，添加到消息中
                if (images.length > 0) {
                    userInputMessage.images = images;
                    logger.info(`[Kiro] Kept ${images.length} image(s) in recent history message (distance from end: ${distanceFromEnd})`);
                }
                
                // 如果有被替换的图片，添加占位符说明
                if (imageCount > 0) {
                    const imagePlaceholder = `[此消息包含 ${imageCount} 张图片，已在历史记录中省略]`;
                    userInputMessage.content = userInputMessage.content
                        ? `${userInputMessage.content}\n${imagePlaceholder}`
                        : imagePlaceholder;
                    logger.info(`[Kiro] Replaced ${imageCount} image(s) with placeholder in old history message (distance from end: ${distanceFromEnd})`);
                }
                
                if (toolResults.length > 0) {
                    // 去重 toolResults - Kiro API 不接受重复的 toolUseId
                    const uniqueToolResults = [];
                    const seenIds = new Set();
                    for (const tr of toolResults) {
                        if (!seenIds.has(tr.toolUseId)) {
                            seenIds.add(tr.toolUseId);
                            uniqueToolResults.push(tr);
                        }
                    }
                    userInputMessage.userInputMessageContext = { toolResults: uniqueToolResults };
                }
                
                history.push({ userInputMessage });
            } else if (message.role === 'assistant') {
                let assistantResponseMessage = {
                    content: ''
                };
                let toolUses = [];
                let thinkingText = '';
                
                if (Array.isArray(message.content)) {
                    for (const part of message.content) {
                        if (part.type === 'text') {
                            assistantResponseMessage.content += part.text;
                        } else if (part.type === 'thinking') {
                            thinkingText += (part.thinking ?? part.text ?? '');
                        } else if (part.type === 'tool_use') {
                            if (hasToolDefinitions) {
                                if (this._isEmptyToolInput(part.input)) {
                                    downgradedToolUseIds.add(part.id);
                                    assistantResponseMessage.content += this._formatToolUseAsText(part);
                                } else {
                                    validToolUseIds.add(part.id);
                                    toolUses.push({
                                        input: this._sanitizeToolInput(part.input),
                                        name: toolNameMaps.toKiroName(part.name),
                                        toolUseId: part.id
                                    });
                                }
                            } else {
                                assistantResponseMessage.content += this._formatToolUseAsText(part);
                            }
                        }
                    }
                } else {
                    assistantResponseMessage.content = this.getContentText(message);
                }

                if (thinkingText) {
                    assistantResponseMessage.content = assistantResponseMessage.content
                        ? `${KIRO_THINKING.START_TAG}${thinkingText}${KIRO_THINKING.END_TAG}\n\n${assistantResponseMessage.content}`
                        : `${KIRO_THINKING.START_TAG}${thinkingText}${KIRO_THINKING.END_TAG}`;
                }

                // 只添加非空字段
                if (toolUses.length > 0) {
                    assistantResponseMessage.toolUses = toolUses;
                }

                history.push({ assistantResponseMessage });
            }
        }

        // Build current message
        let currentMessage = processedMessages[processedMessages.length - 1];
        let currentContent = '';
        let currentToolResults = [];
        let currentToolUses = [];
        let currentImages = [];

        // 如果最后一条消息是 assistant，需要将其加入 history，然后创建一个 user 类型的 currentMessage
        // 因为 CodeWhisperer API 的 currentMessage 必须是 userInputMessage 类型
        if (currentMessage.role === 'assistant') {
            logger.info('[Kiro] Last message is assistant, moving it to history and creating user currentMessage');
            
            // 构建 assistant 消息并加入 history
            let assistantResponseMessage = {
                content: '',
                toolUses: []
            };
            let thinkingText = '';
            if (Array.isArray(currentMessage.content)) {
                for (const part of currentMessage.content) {
                    if (part.type === 'text') {
                        assistantResponseMessage.content += part.text;
                    } else if (part.type === 'thinking') {
                        thinkingText += (part.thinking ?? part.text ?? '');
                    } else if (part.type === 'tool_use') {
                        if (hasToolDefinitions) {
                            if (this._isEmptyToolInput(part.input)) {
                                downgradedToolUseIds.add(part.id);
                                assistantResponseMessage.content += this._formatToolUseAsText(part);
                            } else {
                                validToolUseIds.add(part.id);
                                assistantResponseMessage.toolUses.push({
                                    input: this._sanitizeToolInput(part.input),
                                    name: toolNameMaps.toKiroName(part.name),
                                    toolUseId: part.id
                                });
                            }
                        } else {
                            assistantResponseMessage.content += this._formatToolUseAsText(part);
                        }
                    }
                }
            } else {
                assistantResponseMessage.content = this.getContentText(currentMessage);
            }
            if (thinkingText) {
                assistantResponseMessage.content = assistantResponseMessage.content
                    ? `${KIRO_THINKING.START_TAG}${thinkingText}${KIRO_THINKING.END_TAG}\n\n${assistantResponseMessage.content}`
                    : `${KIRO_THINKING.START_TAG}${thinkingText}${KIRO_THINKING.END_TAG}`;
            }
            if (assistantResponseMessage.toolUses.length === 0) {
                delete assistantResponseMessage.toolUses;
            }
            history.push({ assistantResponseMessage });

            // 设置 currentContent 为 "Continue"，因为我们需要一个 user 消息来触发 AI 继续
            currentContent = 'Continue';

            // tool_choice 强约束在最后 user 消息（Continue）末尾追加，保持遵循度
            if (toolChoiceInstruction) {
                currentContent = `${currentContent}\n\n${toolChoiceInstruction}`;
            }
            if (toolReliabilityInstruction) {
                currentContent = `${currentContent}\n\n${toolReliabilityInstruction}`;
            }
        } else {
            // 最后一条消息是 user，需要确保 history 最后一个元素是 assistantResponseMessage
            // Kiro API 要求 history 必须以 assistantResponseMessage 结尾
            if (history.length > 0) {
                const lastHistoryItem = history[history.length - 1];
                if (!lastHistoryItem.assistantResponseMessage) {
                    // 补一个最小占位。用单个点号（.）而不是 "(acknowledged)" 这类描述性文本——
                    // 模型会把 "(acknowledged)" 当成自己过去的回复风格并延续。
                    // 单点号满足 API 非空校验，且不会被模型当作有意义的回复。
                    logger.info('[Kiro] History does not end with assistantResponseMessage, adding minimal placeholder');
                    history.push({
                        assistantResponseMessage: {
                            content: '.'
                        }
                    });
                }
            }
            
            // 处理 user 消息
            if (Array.isArray(currentMessage.content)) {
                for (const part of currentMessage.content) {
                    if (part.type === 'text') {
                        currentContent += part.text;
                    } else if (part.type === 'tool_result') {
                        if (hasToolDefinitions && validToolUseIds.has(part.tool_use_id) && !downgradedToolUseIds.has(part.tool_use_id)) {
                            currentToolResults.push(this._buildKiroToolResult(part));
                        } else {
                            currentContent += this._formatToolResultAsText(part);
                        }
                    } else if (part.type === 'tool_use') {
                        if (hasToolDefinitions) {
                            if (this._isEmptyToolInput(part.input)) {
                                downgradedToolUseIds.add(part.id);
                                currentContent += this._formatToolUseAsText(part);
                            } else {
                                validToolUseIds.add(part.id);
                                currentToolUses.push({
                                    input: this._sanitizeToolInput(part.input),
                                    name: toolNameMaps.toKiroName(part.name),
                                    toolUseId: part.id
                                });
                            }
                        } else {
                            currentContent += this._formatToolUseAsText(part);
                        }
                    } else if (part.type === 'document') {
                        // Handle PDF / text documents by parsing to inline text
                        const mediaType = part?.source?.media_type || '';
                        if (mediaType === 'application/pdf' && part?.source?.type === 'base64' && part?.source?.data) {
                            const pdfText = await parsePdfBase64ToText(part.source.data);
                            if (pdfText) {
                                const label = part?.title ? `PDF Document: ${part.title}` : 'PDF Document';
                                currentContent += `\n\n[${label}]\n${pdfText}\n[End of Document]\n\n`;
                                logger.info(`[Kiro] Parsed PDF in current message (${pdfText.length} chars)`);
                            } else {
                                currentContent += `\n[PDF document could not be parsed]\n`;
                            }
                        } else if (part?.source?.type === 'text' && part?.source?.data) {
                            currentContent += `\n\n[Document]\n${part.source.data}\n[End of Document]\n\n`;
                        } else if (part?.source?.type === 'url' && part?.source?.url) {
                            const label = part?.title ? `Document: ${part.title}` : 'Document';
                            currentContent += `\n\n[${label}]\n[Source URL: ${part.source.url}]\n[End of Document]\n\n`;
                        }
                    } else if (part.type === 'image') {
                        currentImages.push({
                            format: part.source.media_type.split('/')[1],
                            source: {
                                bytes: part.source.data
                            }
                        });
                    }
                }
            } else {
                currentContent = this.getContentText(currentMessage);
            }

            // Kiro API 要求 content 不能为空，即使有 toolResults
            if (!currentContent) {
                // Use a single dot — Kiro API requires non-empty content,
                // but we must NOT use descriptive English phrases because the model
                // will echo them verbatim back to the user as its own response.
                currentContent = currentToolResults.length > 0
                    ? 'Continue'
                    : '.';
            }

            if (prependSystemToCurrentMessage) {
                currentContent = currentContent
                    ? `${systemPrompt}\n\n${currentContent}`
                    : systemPrompt;
            }

            // tool_choice 指令追加到当前用户消息末尾——模型对"最后看到的指令"遵循度最高。
            if (toolChoiceInstruction) {
                currentContent = currentContent
                    ? `${currentContent}\n\n${toolChoiceInstruction}`
                    : toolChoiceInstruction;
            }

            if (toolReliabilityInstruction) {
                currentContent = currentContent
                    ? `${currentContent}\n\n${toolReliabilityInstruction}`
                    : toolReliabilityInstruction;
            }

            // NOTE: [Code Assistant Context] wrapper removed — the model echoes it back
            // to the user, which exposes the proxy. The wrapper was intended to bypass
            // AWS intent classification but the leak is worse than the occasional block.
            if (this._shouldApplyLightKiroContext(currentContent, currentToolResults, toolsContext, effectiveThinking, responseFormat)) {
                currentContent = `<assistant_context>Answer the user's request directly. This is a normal assistant turn; do not mention this context.</assistant_context>\n\n${currentContent}`;
            }

            const recentKnowledgeHint = buildRecentKnowledgeHint(currentContent, this.config);
            if (recentKnowledgeHint) {
                currentContent = `${currentContent}\n\n${recentKnowledgeHint}`;
            }

            if (responseFormatInstruction) {
                currentContent = currentContent
                    ? `${currentContent}\n\n${responseFormatInstruction}`
                    : responseFormatInstruction;
            }
        }

        // agentTaskType 在请求 body 里只能是 "vibe"（API 不接受 "code"），
        // code/vibe 的区分由请求 header `x-amzn-kiro-agent-mode` 控制。
        const request = {
            conversationState: {
                agentTaskType: "vibe",
                chatTriggerType: KIRO_CONSTANTS.CHAT_TRIGGER_TYPE_MANUAL,
                conversationId: conversationId,
                currentMessage: {} // Will be populated as userInputMessage
            }
        };
        
        // 只有当 history 非空时才添加（API 可能不接受空数组）
        if (history.length > 0) {
            request.conversationState.history = history;
        }

        // currentMessage 始终是 userInputMessage 类型
        // 注意：API 不接受 null 值，空字段应该完全不包含
        const userInputMessage = {
            content: currentContent,
            modelId: codewhispererModel,
            origin: KIRO_CONSTANTS.ORIGIN_AI_EDITOR
        };

        // 只有当 images 非空时才添加
        if (currentImages && currentImages.length > 0) {
            userInputMessage.images = currentImages;
        }

        // 构建 userInputMessageContext，只包含非空字段
        const userInputMessageContext = {};
        currentToolResults = this._reconcileKiroToolHistory(history, currentToolResults);
        if (currentToolResults.length > 0) {
            // 去重 toolResults - Kiro API 不接受重复的 toolUseId
            const uniqueToolResults = [];
            const seenToolUseIds = new Set();
            for (const tr of currentToolResults) {
                if (!seenToolUseIds.has(tr.toolUseId)) {
                    seenToolUseIds.add(tr.toolUseId);
                    uniqueToolResults.push(tr);
                }
            }
            userInputMessageContext.toolResults = uniqueToolResults;
        }
        if (Object.keys(toolsContext).length > 0 && toolsContext.tools) {
            userInputMessageContext.tools = toolsContext.tools;
        }

        // 只有当 userInputMessageContext 有内容时才添加
        if (Object.keys(userInputMessageContext).length > 0) {
            userInputMessage.userInputMessageContext = userInputMessageContext;
        }

        request.conversationState.currentMessage.userInputMessage = userInputMessage;

        if (this.authMethod === KIRO_CONSTANTS.AUTH_METHOD_SOCIAL) {
            request.profileArn = this.profileArn;
        }

        Object.defineProperty(request, '_kiroToolNameMaps', {
            value: toolNameMaps,
            enumerable: false
        });
        Object.defineProperty(request, '_kiroOriginalTools', {
            value: tools || [],
            enumerable: false
        });

        // 监控钩子：内部请求转换
        const monitorRequestId = requestContext?._monitorRequestId;
        if (monitorRequestId) {
            try {
                const { getPluginManager } = await import('../../core/plugin-manager.js');
                const pluginManager = getPluginManager();
                if (pluginManager) {
                    await pluginManager.executeHook('onInternalRequestConverted', {
                        requestId: monitorRequestId,
                        internalRequest: request,
                        converterName: 'buildCodewhispererRequest'
                    });
                }
            } catch (e) {
                logger.error('[Kiro] Error calling onInternalRequestConverted hook:', e.message);
            }
        }

        return request;
    }

    parseEventStreamChunk(rawData, toolNameMaps = null) {
        const rawBuffer = toBuffer(rawData);
        const rawStr = rawBuffer.toString('utf8');
        let fullContent = '';
        const toolCalls = [];

        // 复用流式解析器：brace-counting 会枚举整条响应里所有 JSON payload，
        // 不像旧的正则 "break-on-first-match" 那样漏掉同一事件块里的后续对象；
        // 同时原生支持 toolUseInput / toolUseStop 续传事件，避免参数被截断。
        let { events, recognized } = this.parseAwsEventStreamBuffer(rawBuffer);
        if (!recognized && events.length === 0) {
            events = this.parseAwsEventStreamBuffer(rawStr).events;
        }

        const pendingByToolUseId = new Map();
        const orderedToolUseIds = [];

        const ensurePending = (toolUseId, name) => {
            if (!pendingByToolUseId.has(toolUseId)) {
                pendingByToolUseId.set(toolUseId, {
                    id: toolUseId,
                    type: "function",
                    function: {
                        name: toolNameMaps?.fromKiroName ? toolNameMaps.fromKiroName(name) : name,
                        arguments: ""
                    }
                });
                orderedToolUseIds.push(toolUseId);
            }
            return pendingByToolUseId.get(toolUseId);
        };

        const finalizeToolCall = (toolUseId) => {
            const tc = pendingByToolUseId.get(toolUseId);
            if (!tc) return;
            try {
                const args = JSON.parse(tc.function.arguments || '{}');
                tc.function.arguments = JSON.stringify(args);
            } catch (e) {
                logger.warn(`[Kiro] Tool call arguments not valid JSON: ${tc.function.arguments}`);
            }
            toolCalls.push(tc);
            pendingByToolUseId.delete(toolUseId);
        };

        for (const ev of events) {
            if (ev.type === 'error') {
                throw createKiroEventStreamError(ev);
            } else if (ev.type === 'content') {
                if (ev.data) fullContent += ev.data;
            } else if (ev.type === 'toolUse') {
                const { toolUseId, name, input, stop } = ev.data || {};
                if (!toolUseId) continue;
                const tc = ensurePending(toolUseId, name);
                if (input) tc.function.arguments += input;
                if (stop) finalizeToolCall(toolUseId);
            } else if (ev.type === 'toolUseInput') {
                const { toolUseId, input } = ev.data || {};
                if (!input) continue;
                let targetId = toolUseId;
                if (!targetId) {
                    for (let i = orderedToolUseIds.length - 1; i >= 0; i--) {
                        if (pendingByToolUseId.has(orderedToolUseIds[i])) {
                            targetId = orderedToolUseIds[i];
                            break;
                        }
                    }
                }
                if (targetId && pendingByToolUseId.has(targetId)) {
                    pendingByToolUseId.get(targetId).function.arguments += input;
                }
            } else if (ev.type === 'toolUseStop') {
                let targetId = ev.data?.toolUseId;
                if (!targetId) {
                    for (let i = orderedToolUseIds.length - 1; i >= 0; i--) {
                        const id = orderedToolUseIds[i];
                        if (pendingByToolUseId.has(id)) {
                            targetId = id;
                            break;
                        }
                    }
                }
                if (targetId && pendingByToolUseId.has(targetId)) {
                    finalizeToolCall(targetId);
                }
            }
        }

        for (const id of orderedToolUseIds) {
            if (pendingByToolUseId.has(id)) finalizeToolCall(id);
        }

        // 检查解析后文本中的 bracket 格式工具调用
        const bracketToolCalls = parseBracketToolCalls(fullContent);
        if (bracketToolCalls) {
            toolCalls.push(...bracketToolCalls);
            // 从响应文本中移除工具调用文本
            for (const tc of bracketToolCalls) {
                const funcName = tc.function.name;
                const escapedName = funcName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
                const pattern = new RegExp(`\\[Called\\s+${escapedName}\\s+with\\s+args:\\s*\\{[^}]*(?:\\{[^}]*\\}[^}]*)*\\}\\]`, 'gs');
                fullContent = fullContent.replace(pattern, '');
            }
            fullContent = fullContent.trim();
        }

        const uniqueToolCalls = restoreKiroToolCallNames(deduplicateToolCalls(toolCalls), toolNameMaps);
        return { content: fullContent || '', toolCalls: uniqueToolCalls };
    }
 

    /**
     * 调用 API 并处理错误重试
     */
    async callApi(method, model, body, isRetry = false, retryCount = 0, requestContext = {}) {
        if (!this.isInitialized) await this.initialize();
        const maxRetries = this.config.REQUEST_MAX_RETRIES || 3;
        const baseDelay = this.config.REQUEST_BASE_DELAY || 1000; // 1 second base delay

        // 处理不同格式的请求体（messages 或 contents）
        let messages = body.messages;
        if (!messages && body.contents) {
            // 将 Gemini 格式的 contents 转换为 messages 格式
            messages = body.contents.map(content => ({
                role: content.role || 'user',
                content: content.parts?.map(part => part.text).join('') || ''
            }));
        }
        
        if (!messages || !Array.isArray(messages) || messages.length === 0) {
            throw new Error('No messages found in request body');
        }

        const requestData = await this.buildCodewhispererRequest(messages, model, body.tools, body.system, body.thinking, body.tool_choice, body.response_format, requestContext);

        try {
            const token = this.accessToken; // Use the already initialized token
            const hasTools = body.tools && body.tools.length > 0;
            const headers = {
                'Authorization': `Bearer ${token}`,
                'amz-sdk-invocation-id': `${uuidv4()}`,
                'amz-sdk-request': `attempt=${retryCount + 1}; max=3`,
                'x-amzn-kiro-agent-mode': hasTools ? 'code' : 'vibe',
            };

            const requestUrl = getKiroRequestUrl(model, this.baseUrl);
            const axiosConfig = {
                method: 'post',
                url: requestUrl,
                data: requestData,
                headers,
                responseType: 'arraybuffer'
            };
            this._applySidecar(axiosConfig);
            const releaseThrottle = await acquireKiroRequestSlot(this.config);
            let response;
            try {
                response = await this.axiosInstance.request(axiosConfig);
            } finally {
                releaseThrottle();
            }
            response._kiroToolNameMaps = requestData._kiroToolNameMaps;
            response._kiroOriginalTools = requestData._kiroOriginalTools || body.tools || [];
            return response;
        } catch (error) {
            const status = error.response?.status;
            const errorCode = error.code;
            const errorMessage = sanitizeProviderLeakText(error.message || '');
            if (status === 400 && isInvalidKiroModelError(error)) {
                logger.error(`[Kiro] Invalid model ID from upstream. requested=${model}, resolved=${resolveKiroModel(model)}`);
            }
            
            // 检查是否为可重试的网络错误
            const isNetworkError = isRetryableNetworkError(error);
            
            // Handle 401 (Unauthorized) - refresh UUID first, then try to refresh token
            if (status === 401 && !isRetry) {
                logger.info('[Kiro] Received 401. Refreshing UUID and triggering background refresh via PoolManager...');
                
                // 1. 先刷新 UUID
                const newUuid = this._refreshUuid();
                if (newUuid) {
                    logger.info(`[Kiro] UUID refreshed: ${this.uuid} -> ${newUuid}`);
                    this.uuid = newUuid;
                }
                
                // 标记当前凭证为不健康（会自动进入刷新队列）
                this._markCredentialNeedRefresh('401 Unauthorized - Triggering auto-refresh');
                // Mark error for credential switch without recording error count
                error.shouldSwitchCredential = true;
                error.skipErrorCount = true;
                throw error;
            }
    
            // Handle 402 (Payment Required / Quota Exceeded) - verify usage and mark as unhealthy with recovery time
            if (status === 402 && !isRetry) {
                await this._handle402Error(error, 'callApi');
            }

            // Handle 403 (Forbidden). Most Kiro 403s are account/policy/quota/profile issues,
            // not expired access tokens, so do not blindly refresh.
            if (status === 403 && !isRetry) {
                this._handleForbiddenCredentialError(error, 'callApi');
                // Mark error for credential switch without recording error count
                error.shouldSwitchCredential = true;
                error.skipErrorCount = true;
                throw error;
            }
            
            // Handle 429 (Too Many Requests) - respect Retry-After header, then switch credential
            if (status === 429) {
                const retryAfterHeader = error.response?.headers?.['retry-after'];
                let waitMs = baseDelay;
                if (retryAfterHeader) {
                    const retryAfterSec = Number(retryAfterHeader);
                    if (Number.isFinite(retryAfterSec) && retryAfterSec > 0) {
                        waitMs = Math.min(retryAfterSec * 1000, 60000); // Cap at 60s
                    }
                }
                logger.info(`[Kiro] Received 429 (Too Many Requests). Retry-After: ${retryAfterHeader || 'none'}. Waiting ${waitMs}ms before switching credential...`);
                await new Promise(resolve => setTimeout(resolve, waitMs));
                // Mark error for credential switch without recording error count
                error.shouldSwitchCredential = true;
                error.skipErrorCount = true;
                error.circuitBreakerKey = 'kiro_rate_limited';
                throw error;
            }

            // Handle 5xx server errors - wait baseDelay then switch credential
            if (status >= 500 && status < 600) {
                logger.info(`[Kiro] Received ${status} server error. Waiting ${baseDelay}ms before switching credential...`);
                await new Promise(resolve => setTimeout(resolve, baseDelay));
                // Mark error for credential switch without recording error count
                error.shouldSwitchCredential = true;
                error.skipErrorCount = true;
                throw error;
            }

            // Handle network errors (ECONNRESET, ETIMEDOUT, etc.) with exponential backoff
            if (isNetworkError && retryCount < maxRetries) {
                const delay = baseDelay * Math.pow(2, retryCount);
                const errorIdentifier = errorCode || errorMessage.substring(0, 50);
                logger.info(`[Kiro] Network error (${errorIdentifier}). Retrying in ${delay}ms... (attempt ${retryCount + 1}/${maxRetries})`);
                await new Promise(resolve => setTimeout(resolve, delay));
                return this.callApi(method, model, body, isRetry, retryCount + 1, requestContext);
            }

            if (error.response && error.response.data) {
                logger.error('[Kiro] Response body:', this._getErrorResponseText(error).substring(0, 500));
            }
            logger.error(`[Kiro] API call failed (Status: ${status}, Code: ${errorCode}):`, sanitizeProviderLeakText(error.message));
            throw error;
        }
    }

    _getErrorResponseText(error) {
        const data = error?.response?.data;
        if (data === undefined || data === null) {
            return sanitizeProviderLeakText(error?.message || '');
        }
        if (Buffer.isBuffer(data) || data instanceof ArrayBuffer || ArrayBuffer.isView(data)) {
            return sanitizeProviderLeakText(toBuffer(data).toString('utf8'));
        }
        if (typeof data === 'string') {
            return sanitizeProviderLeakText(data);
        }
        try {
            return sanitizeProviderLeakText(JSON.stringify(data));
        } catch {
            return sanitizeProviderLeakText(String(data));
        }
    }

    _isRefreshableForbidden(error) {
        const text = this._getErrorResponseText(error).toLowerCase();
        if (!text) return false;

        const nonRefreshablePatterns = [
            'temporarily is suspended',
            'temporarily suspended',
            'disabled',
            'violation of terms',
            'terms of service',
            'appeal',
            'quota',
            'limit exceeded',
            'payment required',
            'not authorized to access',
            'not allowed'
        ];
        if (nonRefreshablePatterns.some(pattern => text.includes(pattern))) {
            return false;
        }

        const tokenRelated = text.includes('token') ||
            text.includes('authorization') ||
            text.includes('authenticate') ||
            text.includes('credential');
        const refreshableAuthState = text.includes('expired') ||
            text.includes('invalid') ||
            text.includes('unauthorized');

        return tokenRelated && refreshableAuthState;
    }

    _handleForbiddenCredentialError(error, context) {
        const responseText = this._getErrorResponseText(error);
        const responseSnippet = responseText ? responseText.substring(0, 500) : '';

        if (responseSnippet) {
            logger.warn(`[Kiro] 403 response body (${context}): ${responseSnippet}`);
        }

        if (this._isRefreshableForbidden(error)) {
            logger.info(`[Kiro] Received token-related 403 in ${context}. Marking credential as needs refresh.`);
            this._markCredentialNeedRefresh(`403 Forbidden (${context}) - token-related${responseSnippet ? `: ${responseSnippet}` : ''}`, error);
        } else {
            logger.info(`[Kiro] Received non-refreshable 403 in ${context}. Marking credential as unhealthy without refresh.`);
            this._markCredentialUnhealthy(`403 Forbidden (${context})${responseSnippet ? `: ${responseSnippet}` : ''}`, error);
        }
    }

    /**
     * Helper method to refresh the current credential's UUID
     * Used when encountering 401 errors to get a fresh identity
     * @returns {string|null} - The new UUID, or null if refresh failed
     * @private
     */
    _refreshUuid() {
        const poolManager = getProviderPoolManager();
        if (poolManager && this.uuid) {
            const newUuid = poolManager.refreshProviderUuid(this.config.MODEL_PROVIDER || MODEL_PROVIDER.KIRO_API, {
                uuid: this.uuid
            });
            return newUuid;
        } else {
            logger.warn(`[Kiro] Cannot refresh UUID: poolManager=${!!poolManager}, uuid=${this.uuid}`);
            return null;
        }
    }

    /**
     * Helper method to mark the current credential as unhealthy
     * @param {string} reason - The reason for marking unhealthy
     * @param {Error} [error] - Optional error object to attach the marker to
     * @returns {boolean} - Whether the credential was successfully marked as unhealthy
     * @private
     */
    _markCredentialNeedRefresh(reason, error = null) {
        const poolManager = getProviderPoolManager();
        if (poolManager && this.uuid) {
            logger.info(`[Kiro] Marking credential ${this.uuid} as needs refresh. Reason: ${reason}`);
            // 使用新的 markProviderNeedRefresh 方法代替 markProviderUnhealthyImmediately
            poolManager.markProviderNeedRefresh(this.config.MODEL_PROVIDER || MODEL_PROVIDER.KIRO_API, {
                uuid: this.uuid
            });
            // Attach marker to error object to prevent duplicate marking in upper layers
            if (error) {
                error.credentialMarkedUnhealthy = true;
            }
            return true;
        } else {
            logger.warn(`[Kiro] Cannot mark credential as unhealthy: poolManager=${!!poolManager}, uuid=${this.uuid}`);
            return false;
        }
    }
    
    /**
     * Helper method to mark the current credential as unhealthy
     * @param {string} reason - The reason for marking unhealthy
     * @param {Error} [error] - Optional error object to attach the marker to
     * @returns {boolean} - Whether the credential was successfully marked as unhealthy
     * @private
     */
    _markCredentialUnhealthy(reason, error = null) {
        const poolManager = getProviderPoolManager();
        if (poolManager && this.uuid) {
            logger.info(`[Kiro] Marking credential ${this.uuid} as unhealthy. Reason: ${reason}`);
            poolManager.markProviderUnhealthyImmediately(this.config.MODEL_PROVIDER || MODEL_PROVIDER.KIRO_API, {
                uuid: this.uuid
            }, reason);
            // Attach marker to error object to prevent duplicate marking in upper layers
            if (error) {
                error.credentialMarkedUnhealthy = true;
            }
            return true;
        } else {
            logger.warn(`[Kiro] Cannot mark credential as unhealthy: poolManager=${!!poolManager}, uuid=${this.uuid}`);
            return false;
        }
    }

    /**
     * Helper method to mark the current credential as unhealthy with a scheduled recovery time
     * Used for quota exhaustion (402) where quota resets at a specific time (e.g., 1st of next month)
     * @param {string} reason - The reason for marking unhealthy
     * @param {Error} [error] - Optional error object to attach the marker to
     * @param {Date} [recoveryTime] - The time when the credential should be marked healthy again
     * @returns {boolean} - Whether the credential was successfully marked as unhealthy
     * @private
     */
    _markCredentialUnhealthyWithRecovery(reason, error = null, recoveryTime = null) {
        const poolManager = getProviderPoolManager();
        if (poolManager && this.uuid) {
            logger.info(`[Kiro] Marking credential ${this.uuid} as unhealthy with recovery time. Reason: ${reason}, Recovery: ${recoveryTime?.toISOString()}`);
            poolManager.markProviderUnhealthyWithRecoveryTime(this.config.MODEL_PROVIDER || MODEL_PROVIDER.KIRO_API, {
                uuid: this.uuid
            }, reason, recoveryTime);
            // Attach marker to error object to prevent duplicate marking in upper layers
            if (error) {
                error.credentialMarkedUnhealthy = true;
            }
            return true;
        } else {
            logger.warn(`[Kiro] Cannot mark credential as unhealthy: poolManager=${!!poolManager}, uuid=${this.uuid}`);
            return false;
        }
    }

    /**
     * 计算下月1日 00:00:00 UTC 时间
     * @returns {Date} 下月1日的 Date 对象
     * @private
     */
    _getNextMonthFirstDay() {
        const now = new Date();
        return new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth() + 1, 1, 0, 0, 0, 0));
    }

    /**
     * 处理 402 错误（配额耗尽）
     * 验证用量限制并标记凭证为不健康，设置恢复时间为下月1日
     * @param {Error} error - 原始错误对象
     * @param {string} context - 错误发生的上下文（如 'callApi', 'stream'）
     * @throws {Error} 抛出带有切换凭证标记的错误
     * @private
     */
    async _handle402Error(error, context = 'unknown') {
        logger.info(`[Kiro] Received 402 (Quota Exceeded) in ${context}. Verifying usage limits...`);
        try {
            // Verify usage limits to confirm quota exhaustion
            const usageLimits = await this.getUsageLimits();
            const isQuotaExhausted = usageLimits?.usedCount >= usageLimits?.limitCount;
            
            logger.info(`[Kiro] Quota confirmed exhausted: ${usageLimits?.usedCount}/${usageLimits?.limitCount}`);
            // Calculate recovery time: 1st day of next month at 00:00:00 UTC
            const nextMonth = this._getNextMonthFirstDay();
            this._markCredentialUnhealthyWithRecovery('402 Payment Required - Quota Exhausted', error, nextMonth);
        } catch (usageError) {
            logger.warn('[Kiro] Failed to verify usage limits:', usageError.message);
            // If we can't verify, still mark as unhealthy with recovery time
            const nextMonth = this._getNextMonthFirstDay();
            this._markCredentialUnhealthyWithRecovery('402 Payment Required - Quota Exceeded (unverified)', error, nextMonth);
        }
        // Mark error for credential switch without recording error count
        error.shouldSwitchCredential = true;
        error.skipErrorCount = true;
        throw error;
    }

    _processApiResponse(response) {
        const toolNameMaps = response?._kiroToolNameMaps;
        const rawResponseBuffer = toBuffer(response.data);
        const rawResponseText = rawResponseBuffer.toString('utf8');
        //logger.info(`[Kiro] Raw response length: ${rawResponseText.length}`);
        if (rawResponseText.includes("[Called")) {
            logger.info("[Kiro] Raw response contains [Called marker.");
        }

        // 1. Parse structured events and bracket calls from parsed content
        const parsedFromEvents = this.parseEventStreamChunk(rawResponseBuffer, toolNameMaps);
        let fullResponseText = parsedFromEvents.content;
        let allToolCalls = [...parsedFromEvents.toolCalls]; // clone
        //logger.info(`[Kiro] Found ${allToolCalls.length} tool calls from event stream parsing.`);

        // 2. Crucial fix from Python example: Parse bracket tool calls from the original raw response
        const rawBracketToolCalls = allToolCalls.length > 0 ? null : parseBracketToolCalls(rawResponseText);
        if (rawBracketToolCalls) {
            //logger.info(`[Kiro] Found ${rawBracketToolCalls.length} bracket tool calls in raw response.`);
            allToolCalls.push(...restoreKiroToolCallNames(rawBracketToolCalls, toolNameMaps));
        }

        const validationTools = response?._kiroOriginalTools || response?._kiroTools || [];
        // 3. Deduplicate all collected tool calls
        const uniqueToolCalls = this._filterInvalidGeneratedToolCalls(
            deduplicateToolCalls(toOpenAIToolCalls(allToolCalls)),
            validationTools
        );
        //logger.info(`[Kiro] Total unique tool calls after deduplication: ${uniqueToolCalls.length}`);

        // 4. Clean up response text by removing all tool call syntax from the final text.
        // The text from parseEventStreamChunk is already partially cleaned.
        // We re-clean here with all unique tool calls to be certain.
        if (uniqueToolCalls.length > 0) {
            for (const tc of uniqueToolCalls) {
                const funcName = tc.function.name;
                const escapedName = funcName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
                const pattern = new RegExp(`\\[Called\\s+${escapedName}\\s+with\\s+args:\\s*\\{[^}]*(?:\\{[^}]*\\}[^}]*)*\\}\\]`, 'gs');
                fullResponseText = fullResponseText.replace(pattern, '');
            }
            fullResponseText = fullResponseText.trim();
        }
        
        // 5. Final content cleanup: convert escaped newlines to literal newlines
        fullResponseText = sanitizeProviderLeakText(fullResponseText);
        
        //logger.info(`[Kiro] Final response text after tool call cleanup: ${fullResponseText}`);
        //logger.info(`[Kiro] Final tool calls after deduplication: ${JSON.stringify(uniqueToolCalls)}`);
        return { responseText: fullResponseText, toolCalls: uniqueToolCalls };
    }

    /**
     * Normalize incoming requestBody before sending to Kiro:
     * 1. Map output_config.format -> response_format
     * 2. Strip output_config.effort (unsupported, causes errors)
     * 3. Strip thinking.display (unsupported field)
     * 4. Rewrite Claude Code identity strings without dropping tool/workspace instructions
     */
    _normalizeRequestBody(requestBody) {
        // output_config handling
        if (requestBody.output_config) {
            const oc = requestBody.output_config;
            // Map format to response_format if not already set
            if (oc.format && !requestBody.response_format) {
                requestBody.response_format = normalizeResponseFormat(oc.format);
            }
            // Drop output_config entirely — Kiro doesn't understand it
            delete requestBody.output_config;
        }

        if (requestBody.response_format) {
            requestBody.response_format = normalizeResponseFormat(requestBody.response_format);
        }

        // thinking.display is not a standard field — remove it to avoid errors
        if (requestBody.thinking && requestBody.thinking.display !== undefined) {
            delete requestBody.thinking.display;
        }

        if (Array.isArray(requestBody.messages)) {
            for (const message of requestBody.messages) {
                if (!Array.isArray(message?.content)) continue;
                for (const block of message.content) {
                    if (block?.type === 'thinking' && Object.prototype.hasOwnProperty.call(block, 'signature')) {
                        delete block.signature;
                    }
                }
            }
        }

        if (requestBody.thinking?.type && typeof requestBody.thinking.type === 'string') {
            const type = requestBody.thinking.type.toLowerCase().trim();
            if (type === 'disabled' || type === 'off') {
                delete requestBody.thinking;
            }
        }

        // Keep tool/workspace instructions intact, but remove identity strings that
        // can trigger Kiro safety refusals or provider-name leaks.
        if (Array.isArray(requestBody.system)) {
            requestBody.system = requestBody.system.map(entry => rewriteClaudeCodeIdentityInSystemEntry(entry));
            if (requestBody.system.length === 0) {
                delete requestBody.system;
            }
        } else if (typeof requestBody.system === 'string') {
            requestBody.system = rewriteClaudeCodeIdentityText(requestBody.system);
        }

        return requestBody;
    }

    async generateContent(model, requestBody) {
        if (!this.isInitialized) await this.initialize();

        // 临时存储 monitorRequestId
        const requestContext = {
            _monitorRequestId: requestBody._monitorRequestId
        };
        delete requestBody._monitorRequestId;
        if (requestBody._requestBaseUrl) {
            delete requestBody._requestBaseUrl;
        }

        this._normalizeRequestBody(requestBody);
        this._applyEffectiveThinkingToRequestBody(requestBody);

        // 检查 token 是否即将过期，如果是则推送到刷新队列
        if (this.isTokenExpired()) {
            logger.info('[Kiro] Token is expired, refreshing before generateContent...');
            await this.initializeAuth(true);
        } else if (this.isExpiryDateNear()) {
            logger.info('[Kiro] Token is near expiry, marking credential as need refresh...');
            this._markCredentialNeedRefresh('Token near expiry in generateContent');
        }
        
        const finalModel = resolveKiroModel(model);
        logger.info(`[Kiro] Calling generateContent with model: ${finalModel}`);

        // Estimate input tokens before making the API call
        const inputTokens = this.estimateInputTokens(requestBody);

        const maxEmptyRetries = Number(this.config.KIRO_EMPTY_RESPONSE_MAX_RETRIES) || 2;
        let emptyRetryCount = 0;
        let responseText, toolCalls;

        // Retry loop for empty responses
        while (true) {
            const response = await this.callApi('', model, requestBody, false, 0, requestContext);

            try {
                const result = this._processApiResponse(response);
                responseText = result.responseText;
                toolCalls = result.toolCalls;

                // Check for completely empty response
                if (responseText.trim() === '' && toolCalls.length === 0 && emptyRetryCount < maxEmptyRetries) {
                    emptyRetryCount++;
                    logger.warn(`[Kiro] Empty response detected (0 output tokens). Retrying... (attempt ${emptyRetryCount}/${maxEmptyRetries})`);
                    continue;
                }
                break;
            } catch (error) {
         logger.error('[Kiro] Error in generateContent:', error);
                throw error;
            }
        }

        // If still empty after all retries, throw an Anthropic-style transient error
        // so the client sees what looks like a real API hiccup (no leaked proxy text).
        if (responseText.trim() === '' && toolCalls.length === 0) {
            logger.error(`[Kiro] Empty response persisted after ${maxEmptyRetries} retries. Throwing overloaded_error.`);
            const err = new Error('Overloaded');
            err.response = {
                status: 529,
                data: { type: 'error', error: { type: 'overloaded_error', message: 'Overloaded' } }
            };
            err.shouldSwitchCredential = true;
            err.circuitBreakerKey = 'kiro_empty_response';
            throw err;
        }

        const thinkingType = requestBody?.thinking?.type;
        const thinkingRequested = typeof thinkingType === 'string' &&
            (thinkingType.toLowerCase() === 'enabled' || thinkingType.toLowerCase() === 'adaptive');
        const contentForClaude = thinkingRequested
            ? this._toClaudeContentBlocksFromKiroText(responseText)
            : responseText;
        return this.buildClaudeResponse(contentForClaude, false, 'assistant', model, toolCalls, inputTokens, requestBody);
    }

    /**
     * 解析 AWS Event Stream 格式，提取所有完整的 JSON 事件
     * 返回 { events: 解析出的事件数组, remaining: 未处理完的缓冲区 }
     */
    parseAwsEventStreamBuffer(buffer) {
        const binarySource = Buffer.isBuffer(buffer) || buffer instanceof ArrayBuffer || ArrayBuffer.isView(buffer);
        if (binarySource) {
            const sourceBuffer = toBuffer(buffer);
            if (!isPlausibleAwsEventStreamPrelude(sourceBuffer, 0)) {
                return { events: [], remaining: sourceBuffer, recognized: false };
            }

            const parsedFrames = parseAwsEventStreamFrames(sourceBuffer, { recover: false });
            if (parsedFrames.recognized) {
                return parsedFrames;
            }
        }

        const events = [];
        const source = Buffer.isBuffer(buffer) ? buffer.toString('utf8') : String(buffer || '');
        let searchStart = 0;
        let remainingStart = source.length;
        
        while (true) {
            // 查找真正的 JSON payload 起始位置。AWS Event Stream 包含二进制头部，
            // payload 对象里的 key 顺序不稳定，所以不能依赖 {"input": 这类固定开头。
            const jsonStart = source.indexOf('{', searchStart);
            if (jsonStart < 0) {
                remainingStart = source.length;
                break;
            }
            
            // 正确处理嵌套的 {} - 使用括号计数法
            let braceCount = 0;
            let jsonEnd = -1;
            let inString = false;
            let escapeNext = false;
            
            for (let i = jsonStart; i < source.length; i++) {
                const char = source[i];
                
                if (escapeNext) {
                    escapeNext = false;
                    continue;
                }
                
                if (char === '\\') {
                    escapeNext = true;
                    continue;
                }
                
                if (char === '"') {
                    inString = !inString;
                    continue;
                }
                
                if (!inString) {
                    if (char === '{') {
                        braceCount++;
                    } else if (char === '}') {
                        braceCount--;
                        if (braceCount === 0) {
                            jsonEnd = i;
                            break;
                        }
                    }
                }
            }
            
            if (jsonEnd < 0) {
                // 不完整的 JSON，保留在缓冲区等待更多数据
                remainingStart = jsonStart;
                break;
            }
            
            const jsonStr = source.substring(jsonStart, jsonEnd + 1);
            try {
                const parsed = JSON.parse(jsonStr);
                // 处理 content 事件
                if (parsed.content !== undefined && !parsed.followupPrompt) {
                    // JSON.parse already decoded provider string escapes.
                    let decodedContent = parsed.content;
                    events.push({ type: 'content', data: decodedContent });
                }
                // 处理结构化工具调用事件 - 开始事件（包含 name 和 toolUseId）
                else if (parsed.name && parsed.toolUseId) {
                    events.push({ 
                        type: 'toolUse', 
                        data: {
                            name: parsed.name,
                            toolUseId: parsed.toolUseId,
                            input: normalizeKiroToolInput(parsed.input),
                            stop: parsed.stop || false
                        }
                    });
                }
                // 处理工具调用的 input 续传事件（可能包含 toolUseId，且 key 顺序不固定）
                else if (parsed.input !== undefined && !parsed.name) {
                    events.push({
                        type: 'toolUseInput',
                        data: {
                            toolUseId: parsed.toolUseId,
                            input: normalizeKiroToolInput(parsed.input)
                        }
                    });
                    if (parsed.stop === true) {
                        events.push({
                            type: 'toolUseStop',
                            data: {
                                toolUseId: parsed.toolUseId,
                                stop: true
                            }
                        });
                    }
                }
                // 处理工具调用的结束事件：必须是 stop=true 的"纯结束"事件，
                // 排除 {stop:false, contextUsagePercentage:...} 这类心跳、
                // 以及 {stop:true, input:"..."} 这类带收尾参数的事件（后者走 toolUseInput+toolUseStop 组合更保险）。
                else if (parsed.stop === true
                    && parsed.contextUsagePercentage === undefined
                    && parsed.input === undefined
                    && parsed.name === undefined
                    && parsed.content === undefined) {
                    events.push({
                        type: 'toolUseStop',
                        data: {
                            toolUseId: parsed.toolUseId,
                            stop: true
                        }
                    });
                }
                // 处理上下文使用百分比事件（最后一条消息）
                else if (parsed.contextUsagePercentage !== undefined) {
                    events.push({
                        type: 'contextUsage',
                        data: {
                            contextUsagePercentage: parsed.contextUsagePercentage
                        }
                    });
                }
            } catch (e) {
                // JSON 解析失败，跳过这个 "{" 继续搜索，避免二进制头部中的偶然字符阻塞后续 payload
                searchStart = jsonStart + 1;
                remainingStart = searchStart;
                continue;
            }
            
            searchStart = jsonEnd + 1;
            remainingStart = searchStart;
            if (searchStart >= source.length) {
                remainingStart = source.length;
                break;
            }
        }
        
        // 如果 searchStart 有进展，截取剩余部分
        const remaining = remainingStart < source.length ? source.substring(remainingStart) : '';
        
        return { events, remaining };
    }

    /**
     * 真正的流式 API 调用 - 使用 responseType: 'stream'
     */
    async * streamApiReal(method, model, body, isRetry = false, retryCount = 0, requestContext = {}) {
        if (!this.isInitialized) await this.initialize();
        const maxRetries = this.config.REQUEST_MAX_RETRIES || 3;
        const baseDelay = this.config.REQUEST_BASE_DELAY || 1000;

        // 处理不同格式的请求体（messages 或 contents）
        let messages = body.messages;
        if (!messages && body.contents) {
            // 将 Gemini 格式的 contents 转换为 messages 格式
            messages = body.contents.map(content => ({
                role: content.role || 'user',
                content: content.parts?.map(part => part.text).join('') || ''
            }));
        }
        
        if (!messages || !Array.isArray(messages) || messages.length === 0) {
            throw new Error('No messages found in request body');
        }

        const requestData = await this.buildCodewhispererRequest(messages, model, body.tools, body.system, body.thinking, body.tool_choice, body.response_format, requestContext);
        const toolNameMaps = requestData._kiroToolNameMaps;

        const token = this.accessToken;
        const hasTools = body.tools && body.tools.length > 0;
        const headers = {
            'Authorization': `Bearer ${token}`,
            'amz-sdk-invocation-id': `${uuidv4()}`,
            'amz-sdk-request': `attempt=${retryCount + 1}; max=3`,
            'x-amzn-kiro-agent-mode': hasTools ? 'code' : 'vibe',
        };

        const requestUrl = getKiroRequestUrl(model, this.baseUrl);

        let stream = null;
        let releaseThrottle = () => {};
        try {
            const configuredStreamTimeout = Number(this.config.KIRO_STREAM_TIMEOUT_MS);
            const streamTimeout = Number.isFinite(configuredStreamTimeout) && configuredStreamTimeout >= 0
                ? configuredStreamTimeout
                : KIRO_CONSTANTS.STREAM_TIMEOUT;
            const axiosConfig = {
                method: 'post',
                url: requestUrl,
                data: requestData,
                headers,
                timeout: streamTimeout,
                responseType: 'stream'
            };
            this._applySidecar(axiosConfig);
            releaseThrottle = await acquireKiroRequestSlot(this.config);
            const response = await this.axiosInstance.request(axiosConfig);

            stream = response.data;
            const decoder = new KiroAwsEventStreamDecoder();
            let textFallbackBuffer = '';
            let lastContentEvent = null;  // 用于检测连续重复的 content 事件
            let lastContentRepeatCount = 0; // 连续重复计数

            for await (const chunk of stream) {
                const parsedStream = decoder.feed(chunk);
                let events = parsedStream.events;
                if (!parsedStream.recognized &&
                    parsedStream.remaining?.length >= AWS_EVENT_STREAM_PRELUDE_SIZE &&
                    !isPlausibleAwsEventStreamPrelude(parsedStream.remaining, 0)) {
                    textFallbackBuffer += parsedStream.remaining.toString('utf8');
                    decoder.buffer = Buffer.alloc(0);
                    const parsedText = this.parseAwsEventStreamBuffer(textFallbackBuffer);
                    events = parsedText.events;
                    textFallbackBuffer = parsedText.remaining;
                }

                // yield 所有事件，但过滤连续完全相同的 content 事件（Kiro API 有时会重复发送）
                // 只跳过连续重复超过 2 次的相同内容，避免误杀合法重复
                for (const event of events) {
                    if (event.type === 'error') {
                        throw createKiroEventStreamError(event);
                    }
                    if (event.type === 'content' && event.data) {
                        // NOTE: followupPrompt events are already filtered in parseAwsEventStreamBuffer
                        // (line ~2289: !parsed.followupPrompt). No additional check needed here.
                        // 检查是否与上一个 content 事件完全相同
                        const shouldDeduplicate = event.data.length >= 8 && !isWhitespaceOnly(event.data);
                        if (shouldDeduplicate && lastContentEvent === event.data) {
                            lastContentRepeatCount++;
                            if (lastContentRepeatCount >= 2) {
                                // 连续 3+ 次完全相同的内容，跳过（大概率是 API 抖动）
                                continue;
                            }
                        } else if (shouldDeduplicate) {
                            lastContentRepeatCount = 0;
                        }
                        if (shouldDeduplicate) {
                            lastContentEvent = event.data;
                        } else {
                            lastContentEvent = null;
                            lastContentRepeatCount = 0;
                        }
                        yield { type: 'content', content: event.data };
                    } else if (event.type === 'toolUse') {
                        const toolUse = {
                            ...event.data,
                            name: toolNameMaps?.fromKiroName ? toolNameMaps.fromKiroName(event.data?.name) : event.data?.name
                        };
                        yield { type: 'toolUse', toolUse };
                    } else if (event.type === 'toolUseInput') {
                        yield {
                            type: 'toolUseInput',
                            toolUseId: event.data.toolUseId,
                            input: event.data.input
                        };
                    } else if (event.type === 'toolUseStop') {
                        yield {
                            type: 'toolUseStop',
                            toolUseId: event.data.toolUseId,
                            stop: event.data.stop
                        };
                    } else if (event.type === 'contextUsage') {
                        yield { type: 'contextUsage', contextUsagePercentage: event.data.contextUsagePercentage };
                    }
                }
            }
            // Flush any remaining buffer data after stream ends (handles split chunks)
            {
                const finishedStream = decoder.finish();
                let remainingEvents = finishedStream.events;
                if (!finishedStream.recognized && finishedStream.remaining?.length) {
                    textFallbackBuffer += finishedStream.remaining.toString('utf8');
                }
                if (textFallbackBuffer) {
                    const parsedText = this.parseAwsEventStreamBuffer(textFallbackBuffer);
                    remainingEvents = remainingEvents.concat(parsedText.events);
                    textFallbackBuffer = parsedText.remaining;
                }
                for (const event of remainingEvents) {
                    if (event.type === 'error') {
                        throw createKiroEventStreamError(event);
                    }
                    if (event.type === 'content' && event.data) {
                        yield { type: 'content', content: event.data };
                    } else if (event.type === 'toolUse') {
                        const toolUse = {
                            ...event.data,
                            name: toolNameMaps?.fromKiroName ? toolNameMaps.fromKiroName(event.data?.name) : event.data?.name
                        };
                        yield { type: 'toolUse', toolUse };
                    } else if (event.type === 'toolUseInput') {
                        yield {
                            type: 'toolUseInput',
                            toolUseId: event.data.toolUseId,
                            input: event.data.input
                        };
                    } else if (event.type === 'toolUseStop') {
                        yield {
                            type: 'toolUseStop',
                            toolUseId: event.data.toolUseId,
                            stop: event.data.stop
                        };
                    } else if (event.type === 'contextUsage') {
                        yield { type: 'contextUsage', contextUsagePercentage: event.data.contextUsagePercentage };
                    }
                }
            }
        } catch (error) {
            // 确保出错时关闭流
            if (stream && typeof stream.destroy === 'function') {
                stream.destroy();
            }
            
            const status = error.response?.status;
            const errorCode = error.code;
            const errorMessage = sanitizeProviderLeakText(error.message || '');
            if (status === 400 && isInvalidKiroModelError(error)) {
                logger.error(`[Kiro] Invalid model ID from upstream in stream. requested=${model}, resolved=${resolveKiroModel(model)}`);
            }
            
            // 检查是否为可重试的网络错误
            const isNetworkError = isRetryableNetworkError(error);
            
            // Handle 401 (Unauthorized) - try to refresh token first
            if (status === 401 && !isRetry) {
                logger.info('[Kiro] Received 401 in stream. Triggering background refresh via PoolManager...');
                
                // 1. 先刷新 UUID
                const newUuid = this._refreshUuid();
                if (newUuid) {
                    logger.info(`[Kiro] UUID refreshed: ${this.uuid} -> ${newUuid}`);
                    this.uuid = newUuid;
                }
                // 标记当前凭证为不健康（会自动进入刷新队列）
                this._markCredentialNeedRefresh('401 Unauthorized in stream - Triggering auto-refresh');
                // Mark error for credential switch without recording error count
                error.shouldSwitchCredential = true;
                error.skipErrorCount = true;
                throw error;
            }
            
            // Handle 402 (Payment Required / Quota Exceeded) - verify usage and mark as unhealthy with recovery time
            if (status === 402 && !isRetry) {
                await this._handle402Error(error, 'stream');
            }

            // Handle 403 (Forbidden). Most Kiro 403s are account/policy/quota/profile issues,
            // not expired access tokens, so do not blindly refresh.
            if (status === 403 && !isRetry) {
                this._handleForbiddenCredentialError(error, 'stream');
                // Mark error for credential switch without recording error count
                error.shouldSwitchCredential = true;
                error.skipErrorCount = true;
                throw error;
            }
            
            // Handle 429 (Too Many Requests) - respect Retry-After, then switch credential
            if (status === 429) {
                const retryAfterHeader = error.response?.headers?.['retry-after'];
                let waitMs = baseDelay;
                if (retryAfterHeader) {
                    const retryAfterSec = Number(retryAfterHeader);
                    if (Number.isFinite(retryAfterSec) && retryAfterSec > 0) {
                        waitMs = Math.min(retryAfterSec * 1000, 60000);
                    }
                }
                logger.info(`[Kiro] Received 429 (Too Many Requests) in stream. Retry-After: ${retryAfterHeader || 'none'}. Waiting ${waitMs}ms before switching credential...`);
                await new Promise(resolve => setTimeout(resolve, waitMs));
                // Mark error for credential switch without recording error count
                error.shouldSwitchCredential = true;
                error.skipErrorCount = true;
                error.circuitBreakerKey = 'kiro_rate_limited';
                throw error;
            }

            // Handle 5xx server errors - wait baseDelay then switch credential
            if (status >= 500 && status < 600) {
                logger.info(`[Kiro] Received ${status} server error in stream. Waiting ${baseDelay}ms before switching credential...`);
                await new Promise(resolve => setTimeout(resolve, baseDelay));
                // Mark error for credential switch without recording error count
                error.shouldSwitchCredential = true;
                error.skipErrorCount = true;
                throw error;
            }

            // Handle network errors (ECONNRESET, ETIMEDOUT, etc.) with exponential backoff
            if (isNetworkError && retryCount < maxRetries) {
                const delay = baseDelay * Math.pow(2, retryCount);
                const errorIdentifier = errorCode || errorMessage.substring(0, 50);
                logger.info(`[Kiro] Network error (${errorIdentifier}) in stream. Retrying in ${delay}ms... (attempt ${retryCount + 1}/${maxRetries})`);
                await new Promise(resolve => setTimeout(resolve, delay));
                yield* this.streamApiReal(method, model, body, isRetry, retryCount + 1, requestContext);
                return;
            }

            logger.error(`[Kiro] Stream API call failed (Status: ${status}, Code: ${errorCode}):`,  sanitizeProviderLeakText(error.message));
            throw error;
        } finally {
            releaseThrottle();
            // 确保流被关闭，释放资源
            if (stream && typeof stream.destroy === 'function') {
                stream.destroy();
            }
        }
    }

    // 保留旧的非流式方法用于 generateContent
    async streamApi(method, model, body, isRetry = false, retryCount = 0) {
        try {
            return await this.callApi(method, model, body, isRetry, retryCount);
        } catch (error) {
            logger.error('[Kiro] Error calling API:', error);
            throw error;
        }
    }

    // 真正的流式传输实现
    async * generateContentStream(model, requestBody) {
        if (!this.isInitialized) await this.initialize();

        // 临时存储 monitorRequestId
        const requestContext = {
            _monitorRequestId: requestBody._monitorRequestId
        };
        delete requestBody._monitorRequestId;
        if (requestBody._requestBaseUrl) {
            delete requestBody._requestBaseUrl;
        }

        this._normalizeRequestBody(requestBody);
        this._applyEffectiveThinkingToRequestBody(requestBody);

        // 检查 token 是否即将过期，如果是则推送到刷新队列
        if (this.isTokenExpired()) {
            logger.info('[Kiro] Token is expired, refreshing before generateContentStream...');
            await this.initializeAuth(true);
        } else if (this.isExpiryDateNear()) {
            logger.info('[Kiro] Token is near expiry, marking credential as need refresh...');
            this._markCredentialNeedRefresh('Token near expiry in generateContentStream');
        }
        
        const finalModel = resolveKiroModel(model);
        logger.info(`[Kiro] Calling generateContentStream with model: ${finalModel} (real streaming)`);

        let inputTokens = 0;
        let contextUsagePercentage = null;
        const messageId = generateAnthropicMessageId();

        const thinkingType = requestBody?.thinking?.type;
        const thinkingRequested = typeof thinkingType === 'string' &&
            (thinkingType.toLowerCase() === 'enabled' || thinkingType.toLowerCase() === 'adaptive');

        const streamState = {
            thinkingRequested,
            buffer: '',
            pendingTextBeforeThinking: '',
            inThinking: false,
            thinkingExtracted: false,
            thinkingBlockIndex: null,
            textBlockIndex: null,
            nextBlockIndex: 0,
            stoppedBlocks: new Set(),
            stripThinkingLeadingNewline: false,
            stripTextLeadingNewlinesAfterThinking: false,
            hasVisibleText: false,
            hasThinkingContent: false,
        };

        const ensureBlockStart = (blockType) => {
            if (blockType === 'thinking') {
                if (streamState.thinkingBlockIndex != null) return [];
                const idx = streamState.nextBlockIndex++;
                streamState.thinkingBlockIndex = idx;
                return [{
                    type: "content_block_start",
                    index: idx,
                    content_block: { type: "thinking", thinking: "" }
                }];
            }
            if (blockType === 'text') {
                if (streamState.textBlockIndex != null && !streamState.stoppedBlocks.has(streamState.textBlockIndex)) return [];
                const idx = streamState.nextBlockIndex++;
                streamState.textBlockIndex = idx;
                return [{
                    type: "content_block_start",
                    index: idx,
                    content_block: { type: "text", text: "" }
                }];
            }
            return [];
        };

        const stopBlock = (index) => {
            if (index == null) return [];
            if (streamState.stoppedBlocks.has(index)) return [];
            streamState.stoppedBlocks.add(index);
            return [{ type: "content_block_stop", index }];
        };

        const createTextDeltaEvents = (text) => {
            if (!text) return [];
            if (!isWhitespaceOnly(text)) {
                streamState.hasVisibleText = true;
            }
            const events = [];
            events.push(...ensureBlockStart('text'));
            // Event parsing already decoded string escapes; do not rewrite literal "\n" text.
            const decodedText = sanitizeProviderLeakText(text);
            events.push({
                type: "content_block_delta",
                index: streamState.textBlockIndex,
                delta: { type: "text_delta", text: decodedText }
            });
            return events;
        };

        const createThinkingDeltaEvents = (thinking) => {
            if (thinking) {
                streamState.hasThinkingContent = true;
            }
            const events = [];
            events.push(...ensureBlockStart('thinking'));
            // Event parsing already decoded string escapes; do not rewrite literal "\n" text.
            const decodedThinking = thinking;
            events.push({
                type: "content_block_delta",
                index: streamState.thinkingBlockIndex,
                delta: { type: "thinking_delta", thinking: decodedThinking }
            });
            return events;
        };

        const flushBufferedTextBeforeToolUse = (events) => {
            if (!thinkingRequested) {
                if (streamState.buffer) {
                    events.push(...createTextDeltaEvents(streamState.buffer));
                    streamState.buffer = '';
                }
                return;
            }

            if (streamState.inThinking && streamState.buffer) {
                let endPos = findRealThinkingEndTagAtBufferEnd(streamState.buffer);
                if (endPos === -1) endPos = findRealThinkingEndTagBeforeText(streamState.buffer);
                if (endPos !== -1) {
                    const thinkingPart = streamState.buffer.slice(0, endPos);
                    const remaining = streamState.buffer.slice(endPos + KIRO_THINKING.END_TAG.length).trimStart();
                    if (thinkingPart) events.push(...createThinkingDeltaEvents(thinkingPart));
                    streamState.buffer = '';
                    streamState.inThinking = false;
                    streamState.thinkingExtracted = true;
                    streamState.stripThinkingLeadingNewline = false;
                    streamState.stripTextLeadingNewlinesAfterThinking = false;
                    events.push(...createThinkingDeltaEvents(""));
                    if (streamState.thinkingBlockIndex != null) {
                        events.push({
                            type: "content_block_delta",
                            index: streamState.thinkingBlockIndex,
                            delta: { type: "signature_delta", signature: generateFakeThinkingSignature() }
                        });
                    }
                    events.push(...stopBlock(streamState.thinkingBlockIndex));
                    if (remaining) events.push(...createTextDeltaEvents(remaining));
                }
            }

            if (streamState.inThinking) {
                if (streamState.stripThinkingLeadingNewline) {
                    if (streamState.buffer.startsWith('\r\n')) streamState.buffer = streamState.buffer.slice(2);
                    else if (streamState.buffer.startsWith('\n')) streamState.buffer = streamState.buffer.slice(1);
                    streamState.stripThinkingLeadingNewline = false;
                }
                if (streamState.buffer) events.push(...createThinkingDeltaEvents(streamState.buffer));
                streamState.buffer = '';
                streamState.inThinking = false;
                streamState.thinkingExtracted = true;
                streamState.stripTextLeadingNewlinesAfterThinking = false;
                events.push(...createThinkingDeltaEvents(""));
                if (streamState.thinkingBlockIndex != null) {
                    events.push({
                        type: "content_block_delta",
                        index: streamState.thinkingBlockIndex,
                        delta: { type: "signature_delta", signature: generateFakeThinkingSignature() }
                    });
                }
                events.push(...stopBlock(streamState.thinkingBlockIndex));
            }

            if (!streamState.inThinking && !streamState.thinkingExtracted && streamState.pendingTextBeforeThinking) {
                events.push(...createTextDeltaEvents(streamState.pendingTextBeforeThinking));
                streamState.pendingTextBeforeThinking = '';
            }

            if (streamState.thinkingExtracted && streamState.buffer) {
                let rest = streamState.buffer;
                streamState.buffer = '';
                if (streamState.stripTextLeadingNewlinesAfterThinking) {
                    if (rest.startsWith('\r\n\r\n')) rest = rest.slice(4);
                    else if (rest.startsWith('\n\n')) rest = rest.slice(2);
                    streamState.stripTextLeadingNewlinesAfterThinking = false;
                }
                if (rest) events.push(...createTextDeltaEvents(rest));
            }
        };

        function* pushEvents(events) {
            for (const ev of events) {
                yield ev;
            }
        }

        try {
            let totalContent = '';
            let outputTokens = 0;
            let toolCalls = [];
            let fatalStreamError = null;
            let currentToolCall = null; // 仅作"最近一次活跃调用"指针
            // 新：多工具并发/交错时的 map 累积，避免覆盖丢片段
            let pendingToolCalls = new Map(); // toolUseId -> { toolUseId, name, input }
            let pendingOrder = []; // 插入顺序，用于回落"最近未完成调用"

            // 收尾一个工具调用：解析 input JSON（失败时尝试 repairJson 一次），push 到 toolCalls，
            // 并把 content_block_stop 追加到 outEvents 数组里。
            const finalizeStreamToolCall = (toolUseId, outEvents, sink) => {
                const p = pendingToolCalls.get(toolUseId);
                if (!p) return;
                let parsedInput = p.input;
                try {
                    parsedInput = normalizeToolCallArguments(p.input || '{}');
                } catch (e) {
                    // 常见的 JSON 截断/小瑕疵（尾逗号、未引号键），先 repair 再试一次
                    try {
                        parsedInput = normalizeToolCallArguments(repairJson(p.input || '{}'));
                        logger.warn(`[Kiro Stream] Tool '${p.name}' input repaired via repairJson`);
                    } catch (e2) {
                        const diag = diagnoseJsonTruncation(p.input);
                        logger.warn(`[Kiro Stream] Tool '${p.name}' invalid input: ${diag || e2.message}`);
                        fatalStreamError = createToolCallTruncatedError(p, diag || e2.message);
                        outEvents.push(fatalStreamError);
                        pendingToolCalls.delete(toolUseId);
                        return;
                    }
                }
                const validationError = this._getToolInputValidationError(p.name, parsedInput, requestBody.tools);
                if (validationError) {
                    logger.warn(`[Kiro Stream] Dropping invalid generated tool call '${p.name}': ${validationError}`);
                    pendingToolCalls.delete(toolUseId);
                    return;
                }

                const blockIndex = streamState.nextBlockIndex++;
                const normalizedInputJson = JSON.stringify(parsedInput || {});
                sink.push({
                    toolUseId: p.toolUseId,
                    name: p.name,
                    input: parsedInput
                });
                totalContent += p.name || '';
                totalContent += normalizedInputJson;
                outEvents.push({
                    type: "content_block_start",
                    index: blockIndex,
                    content_block: {
                        type: "tool_use",
                        id: p.toolUseId,
                        name: p.name,
                        input: {}
                    }
                });
                outEvents.push({
                    type: "content_block_delta",
                    index: blockIndex,
                    delta: {
                        type: "input_json_delta",
                        partial_json: normalizedInputJson
                    }
                });
                outEvents.push({ type: "content_block_stop", index: blockIndex });
                pendingToolCalls.delete(toolUseId);
            };

            const estimatedInputTokens = this.estimateInputTokens(requestBody);
            const maxEmptyRetries = Number(this.config.KIRO_EMPTY_RESPONSE_MAX_RETRIES) || 1;
            let emptyRetryCount = 0;

            // 1. 先发送 message_start 事件 (only once, outside retry loop)
            yield {
                type: "message_start",
                message: {
                    id: messageId,
                    type: "message",
                    role: "assistant",
                    model: requestBody?.model || model,
                    usage: {
                        input_tokens: estimatedInputTokens,
                        output_tokens: 0,
                        cache_creation_input_tokens: 0,
                        cache_read_input_tokens: 0
                    },
                    content: []
                }
            };

            // Retry loop for empty responses
            retryLoop: while (true) {
                // Reset state for retry
                if (emptyRetryCount > 0) {
                    totalContent = '';
                    outputTokens = 0;
                    toolCalls = [];
                    fatalStreamError = null;
                    currentToolCall = null;
                    pendingToolCalls = new Map();
                    pendingOrder = [];
                    contextUsagePercentage = null;
                    // Reset stream state
                    streamState.buffer = '';
                    streamState.pendingTextBeforeThinking = '';
                    streamState.inThinking = false;
                    streamState.thinkingExtracted = false;
                    streamState.thinkingBlockIndex = null;
                    streamState.textBlockIndex = null;
                    streamState.nextBlockIndex = 0;
                    streamState.stoppedBlocks = new Set();
                    streamState.stripThinkingLeadingNewline = false;
                    streamState.stripTextLeadingNewlinesAfterThinking = false;
                    streamState.hasVisibleText = false;
                    streamState.hasThinkingContent = false;
                }

            // 2. 流式接收并发送每个 content_block_delta
            for await (const event of this.streamApiReal('', model, requestBody, false, 0, requestContext)) {
                if (event.type === 'contextUsage' && event.contextUsagePercentage) {
                    // 捕获上下文使用百分比（包含输入和输出的总使用量）
                    contextUsagePercentage = event.contextUsagePercentage;
                } else if (event.type === 'content' && event.content) {
                    totalContent += event.content;

                    if (!thinkingRequested) {
                        streamState.buffer += event.content;
                        // 确保不切断转义序列 \\n（如果以 \ 结尾，可能后面跟着 n）
                        if (streamState.buffer.endsWith('\\')) {
                            continue;
                        }
                        yield* pushEvents(createTextDeltaEvents(streamState.buffer));
                        streamState.buffer = '';
                        continue;
                    }

                    streamState.buffer += event.content;
                    const events = [];

                    while (streamState.buffer.length > 0) {
                        if (!streamState.inThinking && !streamState.thinkingExtracted) {
                            const startPos = findRealTag(streamState.buffer, KIRO_THINKING.START_TAG);
                            if (startPos !== -1) {
                                const before = streamState.buffer.slice(0, startPos);
                                const beforeCombined = `${streamState.pendingTextBeforeThinking}${before}`;
                                // Avoid creating meaningless text blocks before thinking.
                                if (beforeCombined && !isWhitespaceOnly(beforeCombined)) {
                                    events.push(...createTextDeltaEvents(beforeCombined));
                                }
                                streamState.pendingTextBeforeThinking = '';

                                streamState.buffer = streamState.buffer.slice(startPos + KIRO_THINKING.START_TAG.length);
                                streamState.inThinking = true;
                                streamState.stripThinkingLeadingNewline = true;
                                continue;
                            }

                            const safeLen = Math.max(0, streamState.buffer.length - KIRO_THINKING.START_TAG.length);
                            if (safeLen > 0) {
                                const safeText = streamState.buffer.slice(0, safeLen);
                                if (safeText) {
                                    if (isWhitespaceOnly(safeText)) {
                                        // Buffer whitespace until we know whether a thinking block appears.
                                        // This prevents a leading text block from being created before thinking.
                                        const maxKeep = 1024;
                                        const remaining = maxKeep - streamState.pendingTextBeforeThinking.length;
                                        if (remaining > 0) {
                                            streamState.pendingTextBeforeThinking += safeText.slice(0, remaining);
                                        }
                                    } else {
                                        const combined = `${streamState.pendingTextBeforeThinking}${safeText}`;
                                        streamState.pendingTextBeforeThinking = '';
                                        events.push(...createTextDeltaEvents(combined));
                                    }
                                }
                                streamState.buffer = streamState.buffer.slice(safeLen);
                            }
                            break;
                        }

                        if (streamState.inThinking) {
                            // Strip a single leading newline after `<thinking>` (may be split across chunks).
                            if (streamState.stripThinkingLeadingNewline) {
                                if (streamState.buffer.startsWith('\r\n')) {
                                    streamState.buffer = streamState.buffer.slice(2);
                                    streamState.stripThinkingLeadingNewline = false;
                                } else if (streamState.buffer.startsWith('\n')) {
                                    streamState.buffer = streamState.buffer.slice(1);
                                    streamState.stripThinkingLeadingNewline = false;
                                } else if (streamState.buffer.length > 0) {
                                    streamState.stripThinkingLeadingNewline = false;
                                }
                            }

                            let endPos = findRealThinkingEndTag(streamState.buffer);
                            if (endPos === -1) endPos = findRealThinkingEndTagAtBufferEnd(streamState.buffer);
                            if (endPos === -1) endPos = findRealThinkingEndTagBeforeText(streamState.buffer);
                            if (endPos !== -1) {
                                const thinkingPart = streamState.buffer.slice(0, endPos);
                                if (thinkingPart) events.push(...createThinkingDeltaEvents(thinkingPart));

                                streamState.buffer = streamState.buffer.slice(endPos + KIRO_THINKING.END_TAG.length);
                                streamState.inThinking = false;
                                streamState.thinkingExtracted = true;
                                streamState.stripThinkingLeadingNewline = false;

                                events.push(...createThinkingDeltaEvents(""));
                                if (streamState.thinkingBlockIndex != null) {
                                    events.push({
                                        type: "content_block_delta",
                                        index: streamState.thinkingBlockIndex,
                                        delta: { type: "signature_delta", signature: generateFakeThinkingSignature() }
                                    });
                                }
                                events.push(...stopBlock(streamState.thinkingBlockIndex));

                                // Strip '\n\n' after the end tag once we switch back to text (may arrive in next chunk).
                                streamState.stripTextLeadingNewlinesAfterThinking = true;
                                continue;
                            }

                            const safeLen = Math.max(0, streamState.buffer.length - KIRO_THINKING.END_TAG.length);
                            if (safeLen > 0) {
                                const safeThinking = streamState.buffer.slice(0, safeLen);
                                if (safeThinking) events.push(...createThinkingDeltaEvents(safeThinking));
                                streamState.buffer = streamState.buffer.slice(safeLen);
                            }
                            break;
                        }

                        if (streamState.thinkingExtracted) {
                            let rest = streamState.buffer;
                            streamState.buffer = '';
                            if (streamState.stripTextLeadingNewlinesAfterThinking) {
                                if (rest.startsWith('\r\n\r\n')) rest = rest.slice(4);
                                else if (rest.startsWith('\n\n')) rest = rest.slice(2);
                                streamState.stripTextLeadingNewlinesAfterThinking = false;
                            }
                            if (rest) events.push(...createTextDeltaEvents(rest));
                            break;
                        }
                    }

                    yield* pushEvents(events);
                } else if (event.type === 'toolUse') {
                    const tc = event.toolUse;
                    const toolEvents = [];
                    const flushToolBoundaryText = () => flushBufferedTextBeforeToolUse(toolEvents);

                    // 统计工具调用的内容到 totalContent（用于 token 计算）
                    // 工具调用事件（包含 name 和 toolUseId）
                    if (tc.name && tc.toolUseId) {
                        flushToolBoundaryText();
                        // 遇到工具调用时，立即关闭文本块，避免前端等待到流结束才看到 content_block_stop
                        toolEvents.push(...stopBlock(streamState.textBlockIndex));

                        // 用 map 维护所有活跃工具调用：避免"多工具交错 / 同一工具多次开始"
                        // 把旧调用覆盖丢片段（read/edit 第一次失败的主因之一）。
                        let pending = pendingToolCalls.get(tc.toolUseId);
                        if (!pending) {
                            pending = {
                                toolUseId: tc.toolUseId,
                                name: tc.name,
                                input: ''
                            };
                            pendingToolCalls.set(tc.toolUseId, pending);
                            pendingOrder.push(tc.toolUseId);

                        }
                        pending.input += tc.input || '';
                        currentToolCall = pending;

                        // 本事件带 stop，立即收尾此工具块
                        if (tc.stop) {
                            finalizeStreamToolCall(tc.toolUseId, toolEvents, toolCalls);
                            if (currentToolCall && currentToolCall.toolUseId === tc.toolUseId) {
                                currentToolCall = null;
                            }
                        }
                    }

                    if (toolEvents.length > 0) {
                        yield* pushEvents(toolEvents);
                        if (fatalStreamError) return;
                    }
                } else if (event.type === 'toolUseInput') {
                    // input 续传：优先按 toolUseId 归属，缺失则落到最近一个未完成调用
                    const inputDelta = normalizeKiroToolInput(event.input);
                    if (!inputDelta) continue;

                    let targetId = event.toolUseId;
                    if (!targetId) {
                        for (let i = pendingOrder.length - 1; i >= 0; i--) {
                            if (pendingToolCalls.has(pendingOrder[i])) {
                                targetId = pendingOrder[i];
                                break;
                            }
                        }
                    }
                    const pending = targetId ? pendingToolCalls.get(targetId) : null;
                    if (pending) {
                        pending.input += inputDelta;
                        currentToolCall = pending;
                    }
                } else if (event.type === 'toolUseStop') {
                    // 工具结束：收尾最近一个未完成调用
                    if (event.stop) {
                        let targetId = event.toolUseId;
                        if (!targetId) {
                            for (let i = pendingOrder.length - 1; i >= 0; i--) {
                                if (pendingToolCalls.has(pendingOrder[i])) {
                                    targetId = pendingOrder[i];
                                    break;
                                }
                            }
                        }
                        if (targetId) {
                            const finalEvents = [];
                            finalizeStreamToolCall(targetId, finalEvents, toolCalls);
                            if (currentToolCall && currentToolCall.toolUseId === targetId) {
                                currentToolCall = null;
                            }
                            if (finalEvents.length > 0) yield* pushEvents(finalEvents);
                            if (fatalStreamError) return;
                        }
                    }
                }
            }

            // 流结束兜底：收尾所有剩余工具调用
            for (const id of [...pendingOrder]) {
                if (pendingToolCalls.has(id)) {
                    const finalEvents = [];
                    finalizeStreamToolCall(id, finalEvents, toolCalls);
                    if (finalEvents.length > 0) yield* pushEvents(finalEvents);
                    if (fatalStreamError) return;
                }
            }
            currentToolCall = null;

            if (thinkingRequested && (streamState.inThinking || streamState.buffer || streamState.pendingTextBeforeThinking)) {
                if (streamState.inThinking) {
                    logger.warn('[Kiro] Incomplete thinking tag at stream end');
                    // Strip a single leading newline after `<thinking>` if we haven't yet.
                    if (streamState.stripThinkingLeadingNewline) {
                        if (streamState.buffer.startsWith('\r\n')) streamState.buffer = streamState.buffer.slice(2);
                        else if (streamState.buffer.startsWith('\n')) streamState.buffer = streamState.buffer.slice(1);
                        streamState.stripThinkingLeadingNewline = false;
                    }
                    yield* pushEvents(createThinkingDeltaEvents(streamState.buffer));
                    streamState.buffer = '';
                    yield* pushEvents(createThinkingDeltaEvents(""));
                    if (streamState.thinkingBlockIndex != null) {
                        yield* pushEvents([{
                            type: "content_block_delta",
                            index: streamState.thinkingBlockIndex,
                            delta: { type: "signature_delta", signature: generateFakeThinkingSignature() }
                        }]);
                    }
                    yield* pushEvents(stopBlock(streamState.thinkingBlockIndex));
                } else if (!streamState.thinkingExtracted) {
                    const remaining = `${streamState.pendingTextBeforeThinking}${streamState.buffer}`;
                    streamState.pendingTextBeforeThinking = '';
                    if (remaining) yield* pushEvents(createTextDeltaEvents(remaining));
                    streamState.buffer = '';
                } else {
                    let remaining = streamState.buffer;
                    streamState.buffer = '';
                    if (streamState.stripTextLeadingNewlinesAfterThinking) {
                        if (remaining.startsWith('\r\n\r\n')) remaining = remaining.slice(4);
                        else if (remaining.startsWith('\n\n')) remaining = remaining.slice(2);
                        streamState.stripTextLeadingNewlinesAfterThinking = false;
                    }
                    if (remaining) yield* pushEvents(createTextDeltaEvents(remaining));
                    streamState.buffer = '';
                }
            } else if (!thinkingRequested && streamState.buffer) {
                // 处理非思考模式下剩余的缓冲区数据
                yield* pushEvents(createTextDeltaEvents(streamState.buffer));
                streamState.buffer = '';
            }

            // --- Empty response detection and retry ---
            const isCompletelyEmpty = !streamState.hasVisibleText &&
                !streamState.hasThinkingContent &&
                toolCalls.length === 0 &&
                totalContent.trim() === '';

            if (isCompletelyEmpty && emptyRetryCount < maxEmptyRetries) {
                emptyRetryCount++;
                logger.warn(`[Kiro Stream] Empty response detected (0 output tokens, end_turn). Retrying... (attempt ${emptyRetryCount}/${maxEmptyRetries})`);
                continue retryLoop;
            }

            // --- End of retry loop ---
            break;
            } // end retryLoop

            // If still completely empty after all retries, inject a fallback message
            const isStillEmpty = !streamState.hasVisibleText &&
                !streamState.hasThinkingContent &&
                toolCalls.length === 0 &&
                totalContent.trim() === '';

            if (isStillEmpty) {
                logger.error(`[Kiro Stream] Empty response persisted after ${maxEmptyRetries} retries. Emitting overloaded_error.`);
                // Emit a proper Anthropic error event instead of user-visible fallback text.
                // The client will see this as a transient API hiccup and may auto-retry.
                yield* pushEvents([{
                    type: 'error',
                    error: { type: 'overloaded_error', message: 'Overloaded' }
                }]);
                return;  // Don't continue to the stop block — we already emitted error
            }

            const emittedOnlyThinking = thinkingRequested &&
                streamState.hasThinkingContent &&
                !streamState.hasVisibleText &&
                toolCalls.length === 0;
            if (emittedOnlyThinking) {
                logger.warn('[Kiro Stream] Thinking-only response received; emitting minimal text block and max_tokens stop_reason');
                yield* pushEvents(createTextDeltaEvents(' '));
            }

            yield* pushEvents(stopBlock(streamState.textBlockIndex));

            // 检查文本内容中的 bracket 格式工具调用
            const bracketToolCalls = parseBracketToolCalls(totalContent);
            if (bracketToolCalls && bracketToolCalls.length > 0) {
                for (const btc of bracketToolCalls) {
                    const streamToolCall = normalizeToolCallForStream(btc);
                    if (streamToolCall) toolCalls.push(streamToolCall);
                }
            }

            // 3. 工具调用在流中实时发送，这里不再批量补发

            // 计算 output tokens
            const contentBlocksForCount = thinkingRequested
                ? this._toClaudeContentBlocksFromKiroText(totalContent)
                : [{ type: "text", text: totalContent }];
            const plainForCount = contentBlocksForCount
                .map(b => (b.type === 'thinking' ? (b.thinking ?? '') : (b.text ?? '')))
                .join('');
            outputTokens = this.countTextTokens(plainForCount);

            for (const tc of toolCalls) {
                const input = tc.input !== undefined
                    ? tc.input
                    : normalizeToolCallArguments(tc.function?.arguments || '{}');
                outputTokens += this.countTextTokens(JSON.stringify(input || {}));
            }

            // 计算 input tokens
            // contextUsagePercentage 是包含输入和输出的总使用量百分比
            // 总 token = TOTAL_CONTEXT_TOKENS * contextUsagePercentage / 100
            // input token = 总 token - output token
            if (contextUsagePercentage !== null && contextUsagePercentage > 0) {
                const contextTokens = getContextTokensForModel(model, this.config, finalModel);
                const totalTokens = Math.round(contextTokens * contextUsagePercentage / 100);
                inputTokens = Math.max(0, totalTokens - outputTokens);
                logger.info(`[Kiro] Token calculation from contextUsagePercentage: total=${totalTokens}, output=${outputTokens}, input=${inputTokens}`);
            } else {
                logger.warn('[Kiro Stream] contextUsagePercentage not received, using estimation');
                inputTokens = estimatedInputTokens;
            }

            // 4. 发送 message_delta 事件
            yield {
                type: "message_delta",
                delta: {
                    stop_reason: toolCalls.length > 0 ? "tool_use" : (emittedOnlyThinking ? "max_tokens" : "end_turn"),
                    stop_sequence: null
                },
                usage: {
                    input_tokens: inputTokens,
                    output_tokens: outputTokens,
                    cache_creation_input_tokens: 0,
                    cache_read_input_tokens: 0
                }
            };

            // 5. 发送 message_stop 事件
            yield { type: "message_stop" };

        } catch (error) {
            logger.error('[Kiro] Error in streaming generation:', error);
            throw error;
        }
    }

    /**
     * Count tokens for a given text using Claude's official tokenizer
     */
    countTextTokens(text) {
        return KiroApiService.countTextTokens(text);
    }

    /**
     * Calculate input tokens from request body using Claude's official tokenizer
     */
    estimateInputTokens(requestBody) {
        return KiroApiService.estimateInputTokens(requestBody);
    }

    /**
     * Build Claude compatible response object
     */
    buildClaudeResponse(content, isStream = false, role = 'assistant', model, toolCalls = null, inputTokens = 0, requestBody = null) {
        // Use the original model name from the request, not the Kiro-mapped internal name
        const originalModel = requestBody?.model || model;
        const messageId = generateAnthropicMessageId();
        const validToolCalls = this._filterInvalidGeneratedToolCalls(toolCalls || [], requestBody?.tools || []);

        if (isStream) {
            // Kiro API is "pseudo-streaming", so we'll send a few events to simulate
            // a full Claude stream, but the content/tool_calls will be sent in one go.
            const events = [];

            // 1. message_start event
            events.push({
                type: "message_start",
                message: {
                    id: messageId,
                    type: "message",
                    role: role,
                    model: originalModel,
                    usage: {
                        input_tokens: inputTokens,
                        output_tokens: 0 // Will be updated in message_delta
                    },
                    content: [] // Content will be streamed via content_block_delta
                }
            });
 
            let totalOutputTokens = 0;
            let stopReason = "end_turn";

            if (content) {
                // If there are tool calls AND content, the content block index should be after tool calls
                const contentBlockIndex = validToolCalls.length > 0 ? validToolCalls.length : 0;

                // 2. content_block_start for text
                events.push({
                    type: "content_block_start",
                    index: contentBlockIndex,
                    content_block: {
                        type: "text",
                        text: "" // Initial empty text
                    }
                });
                // 3. content_block_delta for text
                events.push({
                    type: "content_block_delta",
                    index: contentBlockIndex,
                    delta: {
                        type: "text_delta",
                        text: sanitizeProviderLeakText(content)
                    }
                });
                // 4. content_block_stop
                events.push({
                    type: "content_block_stop",
                    index: contentBlockIndex
                });
                totalOutputTokens += this.countTextTokens(content);
                // If there are tool calls, the stop reason remains "tool_use".
                // If only content, it's "end_turn".
                if (validToolCalls.length === 0) {
                    stopReason = "end_turn";
                }
            }

            if (validToolCalls.length > 0) {
                validToolCalls.forEach((tc, index) => {
                    const inputObject = normalizeToolCallArguments(tc.function.arguments);
                    // 2. content_block_start for each tool_use
                    events.push({
                        type: "content_block_start",
                        index: index,
                        content_block: {
                            type: "tool_use",
                            id: tc.id,
                            name: tc.function.name,
                            input: {} // input is streamed via input_json_delta
                        }
                    });

                    // 3. content_block_delta for each tool_use
                    // Since Kiro is not truly streaming, we send the full arguments as one delta.
                    events.push({
                        type: "content_block_delta",
                        index: index,
                        delta: {
                            type: "input_json_delta",
                            partial_json: JSON.stringify(inputObject)
                        }
                    });

                    // 4. content_block_stop for each tool_use
                    events.push({
                        type: "content_block_stop",
                        index: index
                    });
                    totalOutputTokens += this.countTextTokens(JSON.stringify(inputObject));
                });
                stopReason = "tool_use"; // If there are tool calls, the stop reason is tool_use
            }

            // 5. message_delta with appropriate stop reason
            events.push({
                type: "message_delta",
                delta: {
                    stop_reason: stopReason,
                    stop_sequence: null,
                },
                usage: { output_tokens: totalOutputTokens }
            });

            // 6. message_stop event
            events.push({
                type: "message_stop"
            });

            return events; // Return an array of events for streaming
        } else {
            // Non-streaming response (full message object)
            const contentArray = [];
            let outputTokens = 0;

            // 1) Content blocks (text/thinking) first.
            let hasTextContent = false;
            let hasThinkingContent = false;
            if (Array.isArray(content)) {
                for (const block of content) {
                    if (!block || typeof block !== 'object') continue;
                    if (block.type === 'text' && typeof block.text === 'string') {
                        const sanitizedText = sanitizeProviderLeakText(block.text);
                        contentArray.push({ type: 'text', text: sanitizedText });
                        outputTokens += this.countTextTokens(sanitizedText);
                        if (!isWhitespaceOnly(block.text)) hasTextContent = true;
                    } else if (block.type === 'thinking' && typeof block.thinking === 'string') {
                        contentArray.push({ type: 'thinking', thinking: block.thinking, signature: generateFakeThinkingSignature() });
                        outputTokens += this.countTextTokens(block.thinking);
                        if (block.thinking) hasThinkingContent = true;
                    } else if (typeof block.text === 'string' && block.text) {
                        // Best-effort fallback for unknown blocks carrying plain text.
                        const sanitizedText = sanitizeProviderLeakText(block.text);
                        contentArray.push({ type: 'text', text: sanitizedText });
                        outputTokens += this.countTextTokens(sanitizedText);
                        if (!isWhitespaceOnly(block.text)) hasTextContent = true;
                    }
                }
            } else if (content) {
                const sanitizedContent = sanitizeProviderLeakText(content);
                contentArray.push({ type: "text", text: sanitizedContent });
                outputTokens += this.countTextTokens(sanitizedContent);
                if (!isWhitespaceOnly(content)) hasTextContent = true;
            }

            // 2) Append tool_use blocks (if any).
            let stopReason = "end_turn";
            if (validToolCalls.length > 0) {
                for (const tc of validToolCalls) {
                    const inputObject = normalizeToolCallArguments(tc.function.arguments);
                    contentArray.push({
                        type: "tool_use",
                        id: tc.id,
                        name: tc.function.name,
                        input: inputObject
                    });
                    outputTokens += this.countTextTokens(tc.function.arguments);
                }
                stopReason = "tool_use"; // Set stop_reason to "tool_use" when toolCalls exist
            }

            if (hasThinkingContent && !hasTextContent && validToolCalls.length === 0) {
                contentArray.push({ type: 'text', text: ' ' });
                outputTokens += this.countTextTokens(' ');
                stopReason = "max_tokens";
            }

            return {
                id: messageId,
                type: "message",
                role: role,
                model: originalModel,
                stop_reason: stopReason,
                stop_sequence: null,
                usage: {
                    input_tokens: inputTokens,
                    output_tokens: outputTokens
                },
                content: contentArray
            };
        }
    }

    /**
     * List available models
     */
    async listModels() {
        const configuredModels = Array.isArray(this.config?.supportedModels)
            ? this.config.supportedModels.map(id => String(id).trim()).filter(Boolean)
            : [];
        const models = (configuredModels.length > 0 ? configuredModels : KIRO_MODELS).map(id => ({
            name: id
        }));
        
        return { models: models };
    }

    /**
     * Checks if the token is completely expired (cannot be used at all).
     * @returns {boolean} - True if token is expired, false otherwise.
     */
    isTokenExpired() {
        try {
            if (!this.expiresAt) return true;
            const expirationTime = new Date(this.expiresAt);
            const currentTime = new Date();
            // 给 30 秒缓冲，避免请求过程中过期
            const bufferMs = 30 * 1000;
            return expirationTime.getTime() <= (currentTime.getTime() + bufferMs);
        } catch (error) {
            logger.error(`[Kiro] Error checking token expiry: ${error.message}`);
            return true; // Treat as expired if parsing fails
        }
    }

    /**
     * Checks if the given expiresAt timestamp is within 10 minutes from now (needs refresh soon).
     * @returns {boolean} - True if expiresAt is less than 10 minutes from now, false otherwise.
     */
    isExpiryDateNear() {
        try {
            if (!this.expiresAt) return true;
            const expirationTime = new Date(this.expiresAt);
            if (Number.isNaN(expirationTime.getTime())) return true;
            const nearMinutes = 30;
            const { message, isNearExpiry } = formatExpiryLog('Kiro', expirationTime.getTime(), nearMinutes);
            logger.info(message);
            return isNearExpiry;
        } catch (error) {
            logger.error(`[Kiro] Error checking expiry date: ${this.expiresAt}, Error: ${error.message}`);
            return false; // Treat as expired if parsing fails
        }
    }

    /**
     * 后台异步刷新 token（不阻塞当前请求）
     */
    triggerBackgroundRefresh() {
        logger.info('[Kiro] Background token refresh started...');
        this.initializeAuth(true).then(() => {
            logger.info('[Kiro] Background token refresh completed successfully');
        }).catch((error) => {
            logger.error('[Kiro] Background token refresh failed:', error.message);
            // 后台刷新失败不抛出错误，下次请求会重试
        });
    }

    /**
     * Count tokens for a message request (compatible with Anthropic API)
     * POST /v1/messages/count_tokens
     * @param {Object} requestBody - The request body containing model, messages, system, tools, etc.
     * @returns {Object} { input_tokens: number }
     */
    countTokens(requestBody) {
        return KiroApiService.countTokens(requestBody);
    }

    /**
     * 获取用量限制信息
     * @returns {Promise<Object>} 用量限制信息
     */
    async getUsageLimits() {
        if (!this.isInitialized) await this.initialize();

        // Token 刷新策略：
        // 1. 已过期 → 必须等待刷新
        // 2. 即将过期但还能用 → 后台异步刷新，不阻塞当前请求
        // if (this.isTokenExpired()) {
        //     logger.info('[Kiro] Token is expired, must refresh before getUsageLimits request...');
        //     await this.initializeAuth(true);
        // } else if (this.isExpiryDateNear()) {
        //     logger.info('[Kiro] Token is near expiry, triggering background refresh...');
        //     this.triggerBackgroundRefresh();
        // }
        
        // 内部固定的资源类型
        const resourceType = 'AGENTIC_REQUEST';
        
        // 构建请求 URL
        let usageLimitsUrl = this.baseUrl;
        usageLimitsUrl = usageLimitsUrl.replace('generateAssistantResponse', 'getUsageLimits');
        const params = new URLSearchParams({
            isEmailRequired: 'true',
            origin: KIRO_CONSTANTS.ORIGIN_AI_EDITOR,
            resourceType: resourceType
        });
         if (this.authMethod === KIRO_CONSTANTS.AUTH_METHOD_SOCIAL && this.profileArn) {
            params.append('profileArn', this.profileArn);
        }
        const fullUrl = `${usageLimitsUrl}?${params.toString()}`;

        // 动态生成 headers
        const machineId = generateMachineIdFromConfig({
            uuid: this.uuid,
            profileArn: this.profileArn,
            clientId: this.clientId
        });
        const kiroVersion = this.config.KIRO_VERSION || KIRO_CONSTANTS.KIRO_VERSION;
        const { osName, nodeVersion } = getSystemRuntimeInfo(this.config);

        const headers = {
            'Authorization': `Bearer ${this.accessToken}`,
            'x-amz-user-agent': `aws-sdk-js/1.0.34 KiroIDE-${kiroVersion}-${machineId}`,
            'user-agent': `aws-sdk-js/1.0.34 ua/2.1 os/${osName} lang/js md/electron#${nodeVersion} api/codewhispererstreaming#1.0.34 m/E KiroIDE-${kiroVersion}-${machineId}`,
            'amz-sdk-invocation-id': uuidv4(),
            'amz-sdk-request': 'attempt=1; max=1',
        };

        const axiosConfig = {
            method: 'get',
            url: fullUrl,
            headers
        };
        this._applySidecar(axiosConfig);

        try {
            const response = await this.axiosInstance.request(axiosConfig);
            logger.info('[Kiro] Usage limits fetched successfully');
            return response.data;
        } catch (error) {
            const status = error.response?.status;
            
            // 从响应体中提取错误信息
            let errorMessage = error.message;
            if (error.response?.data) {
                // 尝试从响应体中获取错误描述
                const responseData = error.response.data;
                if (typeof responseData === 'string') {
                    errorMessage = responseData;
                } else if (responseData.message) {
                    errorMessage = responseData.message;
                } else if (responseData.error) {
                    errorMessage = typeof responseData.error === 'string' ? responseData.error : responseData.error.message || JSON.stringify(responseData.error);
                }
            }
            
            // 构建包含状态码和错误描述的错误信息
            const formattedError = status
                ? new Error(`API call failed: ${status} - ${errorMessage}`)
                : new Error(`API call failed: ${errorMessage}`);
            
            // 对于用量查询，401/403 错误直接标记凭证为不健康，不重试
            if (status === 401) {
                logger.info('[Kiro] Received 401 on getUsageLimits. Marking credential as unhealthy (no retry)...');
                this._markCredentialNeedRefresh('401 Unauthorized on usage query', formattedError);
                throw formattedError;
            }
            
            if (status === 403) {
                this._handleForbiddenCredentialError(error, 'usage query');
                formattedError.credentialMarkedUnhealthy = true;
                throw formattedError;
            }
            
            logger.error('[Kiro] Failed to fetch usage limits:', formattedError.message, error);
            throw formattedError;
        }
    }
}
