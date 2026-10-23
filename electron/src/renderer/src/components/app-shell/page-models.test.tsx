import { cleanup, render, screen, within } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { afterEach, expect, it, vi } from 'vitest';
import { PageModels, type PageModelSpec } from './page-models';
import type { LoadedPageModel } from './page-model-info';

const mock = vi.hoisted(() => ({ stage: 'ready', api: vi.fn() }));
vi.mock('@/lib/api/client', () => ({ apiJson: mock.api }));
vi.mock('@/hooks/use-backend-status', () => ({ useBackendStatus: () => ({ stage: mock.stage }) }));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));
vi.mock('@tanstack/react-router', () => ({
  Link: ({ to, params, ...props }: { to: string; params: { family: string } }) => (
    <a href={to.replace('$family', params.family)} {...props} />
  ),
}));
const clients: QueryClient[] = [];
afterEach(() => {
  cleanup();
  clients.forEach((client) => client.clear());
  clients.length = 0;
  mock.stage = 'ready';
  vi.clearAllMocks();
});

function mount(models: readonly PageModelSpec[], loaded: LoadedPageModel[] = []) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: Infinity } },
  });
  clients.push(client);
  client.setQueryData(['engines'], {
    tts: {
      active: 'cosyvoice',
      active_model: null,
      backends: [{ id: 'cosyvoice', display_name: 'CosyVoice', available: true }],
    },
    asr: {
      active: 'faster-whisper',
      active_model: 'Systran/large-v3',
      backends: [{ id: 'faster-whisper', display_name: 'Whisper', available: true }],
    },
  });
  client.setQueryData(['model-catalogue'], {
    models: [
      {
        repo_id: 'FunAudioLLM/CosyVoice3',
        label: 'CosyVoice 3 (multilingual)',
        role: 'TTS',
        engines: ['cosyvoice'],
        installed: true,
      },
    ],
  });
  client.setQueryData(['loaded-models'], { models: loaded });
  client.setQueryData(['sidebar-dictation'], {
    enabled: true,
    available: true,
    model_id: 'sherpa-test',
    model: { id: 'sherpa-test', label: 'Voice input model', installed: true },
    capture: { model_id: null, repo_id: 'Systran/capture-fallback', available: true },
  });
  client.setQueryData(['translation-engines'], {
    active: 'openai',
    engines: [
      { id: 'openai', display_name: 'LLM translation', category: 'llm', installed: true },
      { id: 'argos', display_name: 'Argos', category: 'offline', installed: true },
      { id: 'nllb', display_name: 'NLLB', category: 'offline', installed: true },
    ],
  });
  client.setQueryData(['llm-providers'], {
    active: 'global',
    engine_active: 'off',
    providers: [
      { id: 'global', display_name: 'Global', model: 'global-model', configured: true },
      { id: 'feature', display_name: 'Feature', model: 'feature-model', configured: true },
    ],
  });
  client.setQueryData(['llm-skills'], {
    skills: [
      {
        id: 'dub_translation',
        name_key: 'skill.translation',
        enabled: true,
        ready: true,
        provider_override: 'feature',
      },
    ],
  });
  const view = render(
    <QueryClientProvider client={client}>
      <PageModels models={models} />
    </QueryClientProvider>,
  );
  return { ...view, client };
}

it('links compact selected TTS identity to settings and exposes the full repo in its tooltip', () => {
  mount(['tts']);
  const card = screen.getByRole('link');
  expect(card).toHaveAccessibleName(/engineSidebar\.tts \/ CosyVoice 3/);
  expect(card).toHaveAttribute('href', '/settings/models/tts');
  expect(card).toHaveAttribute('data-model-state', 'selected');
  expect(card.getAttribute('title')).toContain('FunAudioLLM/CosyVoice3');
  expect(mock.api).not.toHaveBeenCalled();
});

it('uses shared runtime inventory updates without starting page-level polling', async () => {
  const { client } = mount(['tts']);
  client.setQueryData(['loaded-models'], {
    models: [
      {
        id: 'active-engine:cosyvoice',
        checkpoint: 'FunAudioLLM/CosyVoice3',
        engine_id: 'cosyvoice',
        is_active_engine: true,
      },
    ],
  });
  await screen.findByText('modelMaintenance.inMemory');
  expect(screen.getByRole('link')).toHaveAttribute('data-model-state', 'loaded');
  expect(mock.api).not.toHaveBeenCalled();
});

it('overrides cached in-memory state while the backend is offline', () => {
  mock.stage = 'stopped';
  mount(
    ['tts'],
    [
      {
        id: 'active-engine:cosyvoice',
        checkpoint: 'FunAudioLLM/CosyVoice3',
        engine_id: 'cosyvoice',
      },
    ],
  );
  expect(screen.getByRole('link')).toHaveAttribute('data-model-state', 'offline');
  expect(screen.getByText('modelMaintenance.offline')).toBeVisible();
  expect(mock.api).not.toHaveBeenCalled();
});

it('shows translation skill provider overrides without claiming a remote model is loaded', () => {
  mount(['translation']);
  expect(screen.getByRole('link')).toHaveTextContent('feature-model');
  expect(screen.getByRole('link')).toHaveAttribute('href', '/settings/models/translation');
  expect(screen.getByRole('link')).toHaveAttribute('data-model-state', 'selected');
  expect(screen.queryByText('global-model')).toBeNull();
});

it('honors per-step translation providers and deduplicates only identical selections', () => {
  mount([
    { family: 'translation', engineId: 'argos' },
    { family: 'translation', engineId: 'nllb' },
    { family: 'translation', engineId: 'argos' },
  ]);
  expect(screen.getAllByRole('link')).toHaveLength(2);
  expect(screen.getByText('Argos')).toBeVisible();
  expect(screen.getByText('NLLB')).toBeVisible();
  expect(screen.queryByText('feature-model')).toBeNull();
});

it('keeps Fast voice input and Accurate ASR identities separate when the mode changes', () => {
  const { rerender, client } = mount(['dictation']);
  expect(within(screen.getByRole('region')).getAllByRole('link')).toHaveLength(1);
  expect(screen.getByText('Voice input model')).toBeVisible();
  rerender(
    <QueryClientProvider client={client}>
      <PageModels models={['asr', 'dictation']} />
    </QueryClientProvider>,
  );
  expect(screen.getAllByRole('link')).toHaveLength(2);
  expect(screen.getByText('large-v3')).toBeVisible();
  expect(screen.getByText('Voice input model')).toBeVisible();
});

it('does not attribute the global LLM to an unreported skill', () => {
  mount([{ family: 'llm', skill: 'missing_skill' }]);
  expect(screen.getByRole('link')).toHaveAttribute('data-model-state', 'unavailable');
  expect(screen.getByText('pageModels.unknownModel')).toBeVisible();
  expect(screen.queryByText('global-model')).toBeNull();
});

it('shows an explicitly selected translation CLI rather than the global provider', () => {
  mount([
    {
      family: 'translation',
      identity: {
        label: 'Claude Code',
        engine: 'claude',
        loaded: false,
        available: true,
      },
    },
  ]);
  expect(screen.getByRole('link')).toHaveTextContent('Claude Code');
  expect(screen.queryByText('feature-model')).toBeNull();
  expect(screen.getByRole('link')).toHaveAttribute('data-model-state', 'selected');
});

it('distinguishes explicitly pinned dictation from the effective call capture model', () => {
  mount(['dictation', { family: 'dictation', routing: 'capture' }]);
  const cards = screen.getAllByRole('link');
  expect(cards).toHaveLength(2);
  expect(cards[0]).toHaveTextContent('Voice input model');
  expect(cards[1]).toHaveTextContent('capture-fallback');
  expect(cards[1]).toHaveAttribute('title', expect.stringContaining('Systran/capture-fallback'));
});

it('does not treat a cached dictation wrapper as a loaded recognizer', () => {
  mount(['dictation'], [{ id: 'capture-asr', checkpoint: 'sherpa-test' }]);
  expect(screen.getByRole('link')).toHaveAttribute('data-model-state', 'selected');
});

it('renders no model strip or extra queries for tools without a model', () => {
  mount([]);
  expect(screen.queryByRole('region')).toBeNull();
  expect(mock.api).not.toHaveBeenCalled();
});
