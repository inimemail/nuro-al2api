import { OpenAIApiService } from './openai-core.js';
import logger from '../../utils/logger.js';

const DEEPSEEK_BASE_URL = 'https://api.deepseek.com';

function normalizeDeepSeekApiKey(apiKey) {
    return String(apiKey || '').trim().replace(/^Bearer\s+/i, '');
}

function normalizeDeepSeekBaseUrl(baseUrl) {
    return String(baseUrl || DEEPSEEK_BASE_URL).trim().replace(/\/+$/, '');
}

function throwReadableDeepSeekError(error) {
    const status = error.response?.status;
    const upstreamMessage = error.response?.data?.error?.message
        || error.response?.data?.message
        || error.message;

    if (status === 401 || status === 403) {
        const readableError = new Error(
            `DeepSeek authentication failed (${status}). 请检查 DeepSeek API Key 是否正确、未过期、未误填 Bearer 前缀，并确认该账号有 API 权限。${upstreamMessage ? ` Upstream: ${upstreamMessage}` : ''}`
        );
        readableError.status = status;
        readableError.cause = error;
        throw readableError;
    }

    throw error;
}

function stringifyContentPart(part) {
    if (part === null || part === undefined) {
        return '';
    }

    if (typeof part === 'string') {
        return part;
    }

    if (typeof part !== 'object') {
        return String(part);
    }

    if (typeof part.text === 'string') {
        return part.text;
    }

    if (part.type === 'image_url') {
        const imageUrl = typeof part.image_url === 'string'
            ? part.image_url
            : part.image_url?.url;
        return imageUrl ? `[Image: ${imageUrl}]` : '[Image]';
    }

    if (part.type === 'input_audio') {
        return `[Audio Input: ${part.input_audio?.format || 'audio'}]`;
    }

    if (part.type === 'tool_result') {
        return typeof part.content === 'string' ? part.content : JSON.stringify(part.content || {});
    }

    if (part.type === 'tool_use') {
        return `[Tool use: ${part.name || 'unknown'}]`;
    }

    return JSON.stringify(part);
}

function normalizeMessageContent(content) {
    if (content === null || content === undefined || typeof content === 'string') {
        return content;
    }

    if (Array.isArray(content)) {
        return content
            .map(stringifyContentPart)
            .filter(Boolean)
            .join('\n');
    }

    return stringifyContentPart(content);
}

export function normalizeDeepSeekChatRequest(requestBody = {}) {
    const normalized = { ...requestBody };

    if (Array.isArray(normalized.messages)) {
        normalized.messages = normalized.messages.map(message => {
            if (!message || typeof message !== 'object') {
                return message;
            }

            const normalizedMessage = { ...message };
            if (Object.prototype.hasOwnProperty.call(normalizedMessage, 'content')) {
                normalizedMessage.content = normalizeMessageContent(normalizedMessage.content);
            }
            return normalizedMessage;
        });
    }

    if (normalized.max_completion_tokens !== undefined) {
        if (normalized.max_tokens === undefined) {
            normalized.max_tokens = normalized.max_completion_tokens;
        }
        delete normalized.max_completion_tokens;
    }

    if (normalized.extra_body && typeof normalized.extra_body === 'object' && !Array.isArray(normalized.extra_body)) {
        Object.entries(normalized.extra_body).forEach(([key, value]) => {
            if (normalized[key] === undefined) {
                normalized[key] = value;
            }
        });
        delete normalized.extra_body;
    }

    return normalized;
}

export class DeepSeekApiService extends OpenAIApiService {
    constructor(config) {
        const apiKey = normalizeDeepSeekApiKey(config.DEEPSEEK_API_KEY);
        if (!apiKey) {
            throw new Error("DeepSeek API Key is required for DeepSeekApiService.");
        }
        // Map DeepSeek credentials into the OpenAI-compatible service shape.
        const deepseekConfig = {
            ...config,
            DEEPSEEK_API_KEY: apiKey,
            OPENAI_API_KEY: apiKey,
            OPENAI_BASE_URL: normalizeDeepSeekBaseUrl(config.DEEPSEEK_BASE_URL),
        };
        super(deepseekConfig);
        logger.info(`[DeepSeek] Initialized with base URL: ${deepseekConfig.OPENAI_BASE_URL}`);
    }

    async generateContent(model, requestBody) {
        try {
            return await super.generateContent(model, normalizeDeepSeekChatRequest(requestBody));
        } catch (error) {
            throwReadableDeepSeekError(error);
        }
    }

    async *generateContentStream(model, requestBody) {
        try {
            yield* super.generateContentStream(model, normalizeDeepSeekChatRequest(requestBody));
        } catch (error) {
            throwReadableDeepSeekError(error);
        }
    }

    async listModels() {
        try {
            const response = await this.axiosInstance.get('/models');
            return response.data;
        } catch (error) {
            const status = error.response?.status;
            logger.error(`[DeepSeek] Error listing models (Status: ${status}):`, error.message);
            return {
                object: 'list',
                data: [
                    { id: 'deepseek-v4-flash', object: 'model', owned_by: 'deepseek' },
                    { id: 'deepseek-v4-pro', object: 'model', owned_by: 'deepseek' },
                ]
            };
        }
    }

    async getUsageLimits() {
        try {
            const response = await this.axiosInstance.get('/user/balance');
            return response.data;
        } catch (error) {
            logger.error('[DeepSeek] Failed to get user balance:', error.message);
            throwReadableDeepSeekError(error);
        }
    }
}
