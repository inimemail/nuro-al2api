import { KiroApiService } from '../src/providers/claude/claude-kiro.js';

jest.mock('../src/utils/proxy-utils.js', () => ({
    configureAxiosProxy: jest.fn(),
    configureTLSSidecar: jest.fn(),
    isTLSSidecarEnabledForProvider: jest.fn(() => false),
}));

jest.mock('../src/services/service-manager.js', () => ({
    getProviderPoolManager: jest.fn(() => null),
}));

function stringHeader(name, value) {
    const nameBuffer = Buffer.from(name, 'utf8');
    const valueBuffer = Buffer.from(value, 'utf8');
    const header = Buffer.alloc(1 + nameBuffer.length + 1 + 2 + valueBuffer.length);
    let offset = 0;
    header[offset++] = nameBuffer.length;
    nameBuffer.copy(header, offset);
    offset += nameBuffer.length;
    header[offset++] = 7;
    header.writeUInt16BE(valueBuffer.length, offset);
    offset += 2;
    valueBuffer.copy(header, offset);
    return header;
}

function awsEventFrame(eventType, payload) {
    const headers = Buffer.concat([
        stringHeader(':message-type', 'event'),
        stringHeader(':event-type', eventType),
    ]);
    const payloadBuffer = Buffer.from(JSON.stringify(payload), 'utf8');
    const totalLength = 12 + headers.length + payloadBuffer.length + 4;
    const frame = Buffer.alloc(totalLength);
    frame.writeUInt32BE(totalLength, 0);
    frame.writeUInt32BE(headers.length, 4);
    headers.copy(frame, 12);
    payloadBuffer.copy(frame, 12 + headers.length);
    return frame;
}

describe('Kiro AWS event-stream parsing', () => {
    test('parses real AWS event-stream frames without scanning binary as text', () => {
        const service = new KiroApiService({});
        const payload = Buffer.concat([
            awsEventFrame('assistantResponseEvent', { content: 'hello ' }),
            awsEventFrame('toolUseEvent', {
                name: 'read_file',
                toolUseId: 'toolu_1',
                input: '{"path":"src/app.js"}',
                stop: true,
            }),
            awsEventFrame('contextUsageEvent', { contextUsagePercentage: 12.5 }),
        ]);

        const parsed = service.parseAwsEventStreamBuffer(payload);

        expect(parsed.recognized).toBe(true);
        expect(parsed.remaining.length).toBe(0);
        expect(parsed.events).toEqual([
            { type: 'content', data: 'hello ' },
            {
                type: 'toolUse',
                data: {
                    name: 'read_file',
                    toolUseId: 'toolu_1',
                    input: '{"path":"src/app.js"}',
                    stop: true,
                },
            },
            { type: 'contextUsage', data: { contextUsagePercentage: 12.5 } },
        ]);
    });

    test('falls back to legacy JSON scanning for non-frame payloads', () => {
        const service = new KiroApiService({});

        const parsed = service.parseAwsEventStreamBuffer('xxx{"content":"legacy"}yyy');

        expect(parsed.events).toEqual([{ type: 'content', data: 'legacy' }]);
        expect(parsed.remaining).toBe('');
    });
});

describe('Kiro context compression', () => {
    test('compacts old messages while preserving the latest user turn', () => {
        const service = new KiroApiService({
            KIRO_CONTEXT_COMPRESSION_THRESHOLD_TOKENS: 1200,
            KIRO_CONTEXT_COMPRESSION_TARGET_TOKENS: 900,
            KIRO_CONTEXT_COMPRESSION_KEEP_RECENT_MESSAGES: 4,
            KIRO_CONTEXT_COMPRESSION_MAX_SUMMARY_TOKENS: 256,
            KIRO_CONTEXT_COMPRESSION_MESSAGE_CHARS: 220,
        });
        const oldText = 'old-context '.repeat(900);
        const currentText = 'please answer using the latest requirement';
        const requestBody = {
            model: 'claude-sonnet-4-5',
            messages: [
                { role: 'user', content: [{ type: 'text', text: `${oldText}A` }] },
                { role: 'assistant', content: [{ type: 'text', text: `${oldText}B` }] },
                { role: 'user', content: [{ type: 'text', text: `${oldText}C` }] },
                { role: 'assistant', content: [{ type: 'text', text: 'recent assistant reply' }] },
                { role: 'user', content: [{ type: 'text', text: 'recent user followup' }] },
                { role: 'assistant', content: [{ type: 'text', text: 'another recent assistant reply' }] },
                { role: 'user', content: [{ type: 'text', text: currentText }] },
            ],
        };
        const beforeTokens = service.estimateInputTokens(requestBody);

        const result = service._compactKiroRequestContextIfNeeded(requestBody, 'claude-sonnet-4-5');
        const afterTokens = service.estimateInputTokens(requestBody);

        expect(result.compressed).toBe(true);
        expect(afterTokens).toBeLessThan(beforeTokens);
        expect(requestBody.messages[0].content[0].text).toContain('[Compressed earlier conversation]');
        expect(requestBody.messages[requestBody.messages.length - 1].content[0].text).toBe(currentText);
        expect(requestBody.messages.some(message =>
            JSON.stringify(message.content).includes(`${oldText}A`)
        )).toBe(false);
    });

    test('does not compact when disabled', () => {
        const service = new KiroApiService({
            KIRO_CONTEXT_COMPRESSION_ENABLED: false,
            KIRO_CONTEXT_COMPRESSION_THRESHOLD_TOKENS: 10,
        });
        const requestBody = {
            messages: [
                { role: 'user', content: 'one '.repeat(100) },
                { role: 'assistant', content: 'two '.repeat(100) },
                { role: 'user', content: 'three '.repeat(100) },
            ],
        };

        const result = service._compactKiroRequestContextIfNeeded(requestBody, 'claude-sonnet-4-5');

        expect(result.compressed).toBe(false);
        expect(requestBody.messages).toHaveLength(3);
    });
});
