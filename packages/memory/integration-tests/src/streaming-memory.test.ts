import { spawn } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import type { UUID } from 'node:crypto';
import { createServer } from 'node:net';
import path from 'node:path';
import { openai } from '@ai-sdk/openai';
import { useChat } from '@ai-sdk/react';
import { Agent } from '@mastra/core/agent';
import { renderHook, act, waitFor } from '@testing-library/react';
import type { Message } from 'ai';
import { JSDOM } from 'jsdom';
import { describe, expect, it, beforeAll, afterAll } from 'vitest';
import { memory, weatherAgent } from './mastra/agents/weather';
import { weatherTool } from './mastra/tools/weather';

// Helper to find an available port
async function getAvailablePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = createServer();
    server.listen(0, () => {
      const { port } = server.address() as { port: number };
      server.close(() => resolve(port));
    });
    server.on('error', reject);
  });
}

// Set up JSDOM environment for React testing
const dom = new JSDOM('<!doctype html><html><body></body></html>', {
  url: 'http://localhost',
  pretendToBeVisual: true,
  resources: 'usable',
});
// @ts-ignore - JSDOM types don't match exactly but this works for testing
global.window = dom.window;
global.document = dom.window.document;
global.navigator = dom.window.navigator;
global.fetch = global.fetch || fetch;

describe('Memory Streaming Tests', () => {
  it('should handle multiple tool calls in memory thread history', async () => {
    // Create agent with memory and tools
    const agent = new Agent({
      name: 'test',
      instructions:
        'You are a weather agent. When asked about weather in any city, use the get_weather tool with the city name as the postal code.',
      model: openai('gpt-4o'),
      memory,
      tools: { get_weather: weatherTool },
    });

    const threadId = randomUUID();
    const resourceId = 'test-resource';

    // First weather check
    const stream1 = await agent.stream('what is the weather in LA?', {
      threadId,
      resourceId,
    });

    // Collect first stream
    const chunks1: string[] = [];
    for await (const chunk of stream1.textStream) {
      chunks1.push(chunk);
    }
    const response1 = chunks1.join('');

    expect(chunks1.length).toBeGreaterThan(0);
    expect(response1).toContain('weather');
    expect(response1).toContain('70 degrees');

    // Second weather check
    const stream2 = await agent.stream('what is the weather in Seattle?', {
      threadId,
      resourceId,
    });

    // Collect second stream
    const chunks2: string[] = [];
    for await (const chunk of stream2.textStream) {
      chunks2.push(chunk);
    }
    const response2 = chunks2.join('');

    expect(chunks2.length).toBeGreaterThan(0);
    expect(response2).toContain('Seattle');
    expect(response2).toContain('weather');
    expect(response2).toContain('70 degrees');
  });

  it('should use experimental_generateMessageId for messages in memory', async () => {
    const agent = new Agent({
      name: 'test-msg-id',
      instructions: 'you are a helpful assistant.',
      model: openai('gpt-4o'),
      memory,
    });

    const threadId = randomUUID();
    const resourceId = 'test-resource-msg-id';
    const customIds: UUID[] = [];

    await agent.generate('Hello, world!', {
      threadId,
      resourceId,
      experimental_generateMessageId: () => {
        const id = randomUUID();
        customIds.push(id);
        return id;
      },
    });

    const { messages } = await agent.getMemory()!.query({ threadId });

    expect(messages).toHaveLength(2);
    expect(messages.length).toBe(customIds.length);
    for (const message of messages) {
      if (!(`id` in message)) {
        throw new Error(`Expected message.id`);
      }
      expect(customIds).contains(message.id);
    }
  });

  describe('should stream via useChat after tool call', () => {
    let mastraServer: ReturnType<typeof spawn>;
    let port: number;
    const threadId = randomUUID();
    const resourceId = 'test-resource';

    beforeAll(async () => {
      port = await getAvailablePort();

      mastraServer = spawn(
        'pnpm',
        [
          path.resolve(import.meta.dirname, `..`, `..`, `..`, `cli`, `dist`, `index.js`),
          'dev',
          '--port',
          port.toString(),
        ],
        {
          stdio: 'pipe',
          detached: true, // Run in a new process group so we can kill it and children
        },
      );

      // Wait for server to be ready
      await new Promise<void>((resolve, reject) => {
        let output = '';
        mastraServer.stdout?.on('data', data => {
          output += data.toString();
          console.log(output);
          if (output.includes('http://localhost:')) {
            resolve();
          }
        });
        mastraServer.stderr?.on('data', data => {
          console.error('Mastra server error:', data.toString());
        });

        setTimeout(() => reject(new Error('Mastra server failed to start')), 10000);
      });
    });

    afterAll(() => {
      // Kill the server and its process group
      if (mastraServer?.pid) {
        try {
          process.kill(-mastraServer.pid, 'SIGTERM');
        } catch (e) {
          console.error('Failed to kill Mastra server:', e);
        }
      }
    });

    it('should stream via useChat after tool call', async () => {
      let error: Error | null = null;
      const { result } = renderHook(() => {
        const chat = useChat({
          api: `http://localhost:${port}/api/agents/test/stream`,
          experimental_prepareRequestBody({ messages }: { messages: Message[]; id: string }) {
            return {
              messages: [messages.at(-1)],
              threadId,
              resourceId,
            };
          },
          onFinish(message) {
            console.log('useChat finished', message.id);
          },
          onError(e) {
            error = e;
            console.error('useChat error:', error);
          },
        });
        return chat;
      });

      let messageCount = 0;
      async function expectResponse({ message, responseContains }: { message: string; responseContains: string[] }) {
        messageCount++;
        await act(async () => {
          await result.current.append({
            role: 'user',
            content: message,
          });
        });
        const responseIndex = messageCount * 2 - 1;
        await waitFor(
          () => {
            expect(error).toBeNull();
            expect(result.current.messages).toHaveLength(messageCount * 2);
            for (const should of responseContains) {
              expect(result.current.messages[responseIndex].content).toContain(should);
            }
          },
          { timeout: 1000 },
        );
      }

      await expectResponse({
        message: 'what is the weather in Los Angeles?',
        responseContains: ['Los Angeles', '70 degrees'],
      });

      await expectResponse({
        message: 'what is the weather in Seattle?',
        responseContains: ['Seattle', '70 degrees'],
      });
    });

    it('should stream useChat with client side tool calling', async () => {
      let error: Error | null = null;
      const threadId = randomUUID();

      await weatherAgent.generate(`hi`, {
        threadId,
        resourceId,
      });
      await weatherAgent.generate(`LA weather`, { threadId, resourceId });

      const initialMessages = (await weatherAgent.getMemory()!.query({ threadId })).uiMessages;
      const state = { clipboard: '' };
      const { result } = renderHook(() => {
        const chat = useChat({
          api: `http://localhost:${port}/api/agents/test/stream`,
          initialMessages,
          experimental_prepareRequestBody({ messages }: { messages: Message[]; id: string }) {
            return {
              messages: [messages.at(-1)],
              threadId,
              resourceId,
            };
          },
          onFinish(message) {
            console.log('useChat finished', message.id);
          },
          onError(e) {
            error = e;
            console.error('useChat error:', error);
          },
          onToolCall: async ({ toolCall }) => {
            console.log(toolCall);
            if (toolCall.toolName === `clipboard`) {
              await new Promise(res => setTimeout(res, 10));
              return state.clipboard;
            }
          },
        });
        return chat;
      });

      async function expectResponse({ message, responseContains }: { message: string; responseContains: string[] }) {
        const messageCountBefore = result.current.messages.length;
        await act(async () => {
          await result.current.append({
            role: 'user',
            content: message,
          });
        });
        const uiMessages = result.current.messages;
        await waitFor(
          () => {
            expect(error).toBeNull();
            expect(result.current.messages.length).toBeGreaterThan(messageCountBefore);
          },
          { timeout: 1000 },
        );

        const latestMessage = uiMessages.at(-1);
        if (!latestMessage) throw new Error(`No latest message`);
        if (
          latestMessage.role === `assistant` &&
          latestMessage.parts.length === 2 &&
          latestMessage.parts[1].type === `tool-invocation`
        ) {
          // client side tool call
          return;
        }
        for (const should of responseContains) {
          let searchString = typeof latestMessage.content === `string` ? latestMessage.content : ``;

          for (const part of latestMessage.parts) {
            if (part.type === `text`) {
              searchString += `\n${part.text}`;
            }
            if (part.type === `tool-invocation`) {
              searchString += `\n${JSON.stringify(part.toolInvocation)}`;
            }
          }

          expect(searchString).toContain(should);
        }
      }

      state.clipboard = `test 1!`;
      await expectResponse({
        message: 'whats in my clipboard?',
        responseContains: [state.clipboard],
      });
      await expectResponse({
        message: 'weather in Las Vegas',
        responseContains: ['Las Vegas', '70 degrees'],
      });
      state.clipboard = `test 2!`;
      await expectResponse({
        message: 'whats in my clipboard?',
        responseContains: [state.clipboard],
      });
      state.clipboard = `test 3!`;
      await expectResponse({
        message: 'whats in my clipboard now?',
        responseContains: [state.clipboard],
      });
    });
  });
});
