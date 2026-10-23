import { expect, it } from 'vitest';
import type { CatalogueModel } from '@/features/settings/model-catalogue-query';
import type { EngineFamilyState } from '@/lib/api/types';
import { compactPageModelName, enginePageModel, llmPageModel } from './page-model-info';

const catalogue: CatalogueModel[] = [
  {
    repo_id: 'FunAudioLLM/CosyVoice3',
    label: 'CosyVoice 3 (multilingual zero-shot)',
    role: 'TTS',
    engines: ['cosyvoice'],
    installed: true,
    supported: true,
    size_gb: 1,
  },
];
const selection: EngineFamilyState = {
  active: 'cosyvoice',
  active_model: null,
  backends: [{ id: 'cosyvoice', display_name: 'CosyVoice', available: true, reason: null }],
};

it('identifies an unambiguous selected model before weights load, without claiming it is resident', () => {
  const identity = enginePageModel('tts', selection, catalogue);
  expect(identity).toMatchObject({
    model: 'FunAudioLLM/CosyVoice3',
    loaded: false,
    available: true,
  });
  expect(compactPageModelName(identity)).toBe('CosyVoice 3');
});

it('never picks the only installed variant when a backend supports several models', () => {
  const ambiguous = [
    { ...catalogue[0], engines: ['mlx-audio'] },
    { ...catalogue[0], repo_id: 'other/variant', engines: ['mlx-audio'], installed: false },
  ];
  expect(
    enginePageModel('tts', { ...selection, active: 'mlx-audio' }, ambiguous).model,
  ).toBeUndefined();
});

it('never treats a live sidecar process as evidence that model weights are loaded', () => {
  const identity = enginePageModel('tts', selection, catalogue, [
    {
      id: 'sidecar:cosyvoice',
      checkpoint: 'cosyvoice',
      engine_id: 'cosyvoice',
      is_active_engine: true,
    },
  ]);
  expect(identity).toMatchObject({ model: 'FunAudioLLM/CosyVoice3', loaded: false });
});

it('does not mark a newly selected TTS model loaded because a previous engine remains resident', () => {
  expect(
    enginePageModel('tts', selection, catalogue, [
      {
        id: 'tts',
        checkpoint: 'k2-fsa/OmniVoice',
        engine_id: 'omnivoice',
        is_active_engine: false,
      },
    ]).loaded,
  ).toBe(false);
});

it('keeps explicit selection and rejects stale model variants resident in the same engine', () => {
  const identity = enginePageModel('tts', { ...selection, active_model: 'custom/new' }, catalogue, [
    { id: 'engine:cosyvoice', checkpoint: 'cosyvoice', engine_id: 'cosyvoice' },
    { id: 'active-engine:cosyvoice', checkpoint: 'custom/old', engine_id: 'cosyvoice' },
  ]);
  expect(identity).toMatchObject({ model: 'custom/new', loaded: false });
});

it('never confuses the dictation cache with the accurate ASR model', () => {
  const asr = { ...selection, active: 'whisperx', active_model: 'Systran/large-v3' };
  expect(
    enginePageModel('asr', asr, [], [{ id: 'capture-asr', checkpoint: 'Systran/large-v3' }]).loaded,
  ).toBe(false);
  expect(
    enginePageModel('asr', asr, [], [{ id: 'asr', checkpoint: 'Systran/large-v3' }]).loaded,
  ).toBe(true);
});

it('does not attribute co-loaded WhisperX weights to another ASR backend using the same repo', () => {
  const asr = { ...selection, active: 'faster-whisper', active_model: 'Systran/large-v3' };
  expect(
    enginePageModel('asr', asr, [], [{ id: 'asr', checkpoint: 'Systran/large-v3' }]).loaded,
  ).toBe(false);
  expect(
    enginePageModel(
      'asr',
      asr,
      [],
      [{ id: 'asr', checkpoint: 'Systran/large-v3', engine_id: 'faster-whisper' }],
    ).loaded,
  ).toBe(true);
});

it('does not infer ASR selection from unrelated resident weights', () => {
  expect(
    enginePageModel(
      'asr',
      { ...selection, active: 'faster-whisper' },
      [],
      [{ id: 'asr', checkpoint: 'old/checkpoint' }],
    ).model,
  ).toBeUndefined();
});

it('clears model identity when a family is switched off even if old weights remain loaded', () => {
  expect(
    enginePageModel('tts', { ...selection, active: 'off' }, catalogue, [
      { id: 'tts', checkpoint: 'k2-fsa/OmniVoice' },
    ]),
  ).toEqual({ loaded: false, available: false, inactive: true });
});

const providers = {
  active: 'default',
  engine_active: 'off',
  providers: [
    { id: 'default', display_name: 'Default', model: 'default-model', configured: true },
    { id: 'override', display_name: 'Override', model: 'feature-model', configured: true },
  ],
};

it('resolves feature-specific provider overrides even when saved global LLM selection is off', () => {
  expect(
    llmPageModel(providers, {
      id: 'slot_fitting',
      name_key: 'skill.fit',
      provider_override: 'override',
      enabled: true,
      ready: true,
    }),
  ).toMatchObject({
    model: 'feature-model',
    engine: 'Override',
    available: true,
    inactive: false,
    loaded: false,
  });
});

it('trusts skill readiness and never describes configured remote LLM models as resident', () => {
  expect(llmPageModel(providers)).toMatchObject({ inactive: true, loaded: false });
  expect(
    llmPageModel(providers, {
      id: 'direction_parse',
      name_key: 'skill.direction',
      provider_override: 'override',
      enabled: true,
      ready: false,
    }),
  ).toMatchObject({ available: false, loaded: false });
});
