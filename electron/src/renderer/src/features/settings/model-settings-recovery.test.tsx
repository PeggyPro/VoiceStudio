import {
  QueryClient,
  QueryClientProvider,
  focusManager,
  notifyManager,
  onlineManager,
} from '@tanstack/react-query';
import { act, cleanup, fireEvent, render, renderHook, screen, within } from '@testing-library/react';
import { createInstance } from 'i18next';
import type { ReactNode } from 'react';
import { I18nextProvider } from 'react-i18next';
import { afterAll, afterEach, beforeAll, beforeEach, expect, it, vi } from 'vitest';
import en from '@/i18n/locales/en.json';
import { useEngines } from '@/hooks/use-engines';
import { ApiError } from '@/lib/api/client';
import type { EnginesResponse } from '@/lib/api/types';
import { queryClient, queryKeys } from '@/lib/query';
import { ModelSettings } from './model-settings';

vi.mock('@tanstack/react-router', () => ({
  Link: ({ children }: { children: ReactNode }) => <a href="#">{children}</a>,
}));

const i18n = createInstance();
const fetchMock = vi.fn<typeof fetch>();
const getEngines = vi.fn<() => Promise<Response>>();
const selectEngine = vi.fn<() => Promise<Response>>();
const clients: QueryClient[] = [];
const engines: EnginesResponse = {
  tts: {
    active: 'ready-tts',
    backends: [
      { id: 'ready-tts', display_name: 'Ready TTS', available: true, reason: null },
      { id: 'other-tts', display_name: 'Other TTS', available: true, reason: null },
    ],
  },
  asr: { active: null, backends: [] },
  llm: { active: null, backends: [] },
};
const catalogue = {
  models: [
    {
      repo_id: 'test/voice-model',
      label: 'Voice model',
      role: 'TTS',
      size_gb: 1,
      installed: false,
      supported: true,
    },
  ],
};

function gatewayError() {
  return new Response(null, { status: 502, statusText: 'Bad Gateway' });
}

function setup() {
  // Keep the production retry policy: a real ApiError(502) must not get an
  // immediate retry that would conceal a missing recovery poll.
  const client = new QueryClient({ defaultOptions: queryClient.getDefaultOptions() });
  clients.push(client);
  const wrapper = ({ children }: { children: ReactNode }) => (
    <I18nextProvider i18n={i18n}>
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    </I18nextProvider>
  );
  return { client, wrapper };
}

async function advance(ms = 0) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

beforeAll(async () => {
  await i18n.init({
    lng: 'en',
    resources: { en: { translation: en } },
    interpolation: { escapeValue: false },
  });
  // Only change notification batching, leaving request/retry/poll scheduling
  // intact so timer assertions measure the real hook's recovery interval.
  notifyManager.setScheduler(queueMicrotask);
});

afterAll(() => {
  notifyManager.setScheduler((callback) => setTimeout(callback, 0));
});

beforeEach(() => {
  vi.useFakeTimers();
  focusManager.setFocused(true);
  onlineManager.setOnline(true);
  getEngines.mockReset().mockImplementation(async () => Response.json(engines));
  selectEngine.mockReset().mockImplementation(async () => gatewayError());
  fetchMock.mockReset().mockImplementation(async (input, init) => {
    const path = new URL(String(input), window.location.href).pathname;
    if (path === '/api/engines' && !init?.method) return getEngines();
    if (path === '/api/engines/select' && init?.method === 'POST') return selectEngine();
    if (path === '/api/models') return Response.json(catalogue);
    if (path === '/api/models/install/status') return Response.json({ jobs: [] });
    if (path === '/api/model/loaded') return Response.json({ models: [], count: 0 });
    throw new Error(`Unexpected request: ${init?.method ?? 'GET'} ${path}`);
  });
  vi.stubGlobal('fetch', fetchMock);
});

afterEach(() => {
  cleanup();
  for (const client of clients.splice(0)) client.clear();
  vi.unstubAllGlobals();
  vi.useRealTimers();
  focusManager.setFocused(undefined);
  onlineManager.setOnline(true);
});

it('recovers an initial 502 on the next poll without immediately retrying or selecting an engine', async () => {
  getEngines.mockImplementationOnce(async () => gatewayError());
  const { client, wrapper } = setup();
  const { result } = renderHook(useEngines, { wrapper });
  await advance();

  expect(result.current.isError).toBe(true);
  expect(result.current.error).toBeInstanceOf(ApiError);
  expect(client.getQueryState(queryKeys.engines)?.error).toMatchObject({ status: 502 });
  await advance(14_999);
  expect(getEngines).toHaveBeenCalledTimes(1);
  await advance(1);

  expect(getEngines).toHaveBeenCalledTimes(2);
  expect(result.current.isError).toBe(false);
  expect(result.current.activeTtsReady).toBe(true);
  await advance(60_000);
  expect(getEngines).toHaveBeenCalledTimes(2);
  expect(selectEngine).not.toHaveBeenCalled();
});

it('keeps polling after a 502 with cached ready engines and stops after recovery', async () => {
  const { client, wrapper } = setup();
  client.setQueryData(queryKeys.engines, engines);
  const { result } = renderHook(useEngines, { wrapper });
  getEngines.mockImplementationOnce(async () => gatewayError());
  await act(() => client.invalidateQueries({ queryKey: queryKeys.engines }));

  expect(result.current.isError).toBe(true);
  expect(result.current.activeTtsReady).toBe(true);
  expect(result.current.data).toEqual(engines);
  await advance(14_999);
  expect(getEngines).toHaveBeenCalledTimes(1);
  await advance(1);

  expect(getEngines).toHaveBeenCalledTimes(2);
  expect(result.current.isError).toBe(false);
  await advance(60_000);
  expect(getEngines).toHaveBeenCalledTimes(2);
  expect(selectEngine).not.toHaveBeenCalled();
});

it('waits for a slow engine list without an error or overlapping HTTP requests', async () => {
  let respond!: (response: Response) => void;
  getEngines.mockImplementationOnce(
    () => new Promise<Response>((resolve) => { respond = resolve; }),
  );
  const { wrapper } = setup();
  const { result } = renderHook(useEngines, { wrapper });
  await advance(60_000);

  expect(result.current.isLoading).toBe(true);
  expect(result.current.isError).toBe(false);
  expect(getEngines).toHaveBeenCalledTimes(1);
  await act(async () => { respond(Response.json(engines)); });
  expect(result.current.activeTtsReady).toBe(true);
  expect(result.current.isLoading).toBe(false);
  expect(selectEngine).not.toHaveBeenCalled();
});

it('describes an initial GET failure as loading and recovers through its retry button', async () => {
  getEngines.mockImplementationOnce(async () => gatewayError());
  const { wrapper } = setup();
  render(<ModelSettings family="tts" />, { wrapper });
  await advance();

  const alert = screen.getByRole('alert');
  expect(alert).toHaveTextContent('Could not load engines. Try again.');
  expect(alert).toHaveTextContent('HTTP 502 Bad Gateway');
  expect(alert).not.toHaveTextContent(en.modelSettings.failed);
  fireEvent.click(within(alert).getByRole('button', { name: en.backend.retry }));
  await advance();

  expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  expect(screen.getByRole('heading', { name: 'Ready TTS' })).toBeVisible();
  expect(screen.getByRole('heading', { name: 'Voice model' })).toBeVisible();
  expect(selectEngine).not.toHaveBeenCalled();
});

it('keeps cached engine rows and model-library search mounted during a refresh error and retry', async () => {
  const { client, wrapper } = setup();
  render(<ModelSettings family="tts" />, { wrapper });
  await advance();
  const row = screen.getByRole('heading', { name: 'Ready TTS' });
  const search = screen.getByRole('searchbox');
  fireEvent.change(search, { target: { value: 'Voice model' } });
  getEngines.mockImplementationOnce(async () => gatewayError());
  await act(() => client.invalidateQueries({ queryKey: queryKeys.engines }));

  expect(screen.getByRole('heading', { name: 'Ready TTS' })).toBe(row);
  expect(screen.getByRole('heading', { name: 'Voice model' })).toBeVisible();
  expect(screen.getByRole('searchbox')).toBe(search);
  expect(search).toHaveValue('Voice model');
  const alert = screen.getByRole('alert');
  expect(alert).toHaveTextContent('Could not load engines. Try again.');
  fireEvent.click(within(alert).getByRole('button', { name: en.backend.retry }));
  await advance();

  expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  expect(screen.getByRole('heading', { name: 'Ready TTS' })).toBe(row);
  expect(screen.getByRole('searchbox')).toBe(search);
  expect(search).toHaveValue('Voice model');
  expect(getEngines).toHaveBeenCalledTimes(3);
  expect(selectEngine).not.toHaveBeenCalled();
});

it('keeps the selection-failure message and content when POST fails without retrying the POST', async () => {
  const { wrapper } = setup();
  render(<ModelSettings family="tts" />, { wrapper });
  await advance();
  fireEvent.click(screen.getByRole('button', { name: en.modelSettings.select }));
  await advance();

  const alert = screen.getByRole('alert');
  expect(alert).toHaveTextContent(en.modelSettings.failed);
  expect(alert).not.toHaveTextContent('Could not load engines. Try again.');
  expect(screen.getByRole('heading', { name: 'Ready TTS' })).toBeVisible();
  expect(screen.getByRole('heading', { name: 'Voice model' })).toBeVisible();
  expect(fetchMock).toHaveBeenCalledWith(
    '/api/engines/select',
    expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({ family: 'tts', backend_id: 'other-tts' }),
    }),
  );
  await advance(60_000);
  expect(selectEngine).toHaveBeenCalledTimes(1);
  expect(getEngines).toHaveBeenCalledTimes(1);
});
