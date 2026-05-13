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
