#공통 설정 모아두는 곳.

PITCH_MIN = 40
PITCH_MAX = 76
REST_TOKEN = 0
VOCAB_SIZE = 128

BARS = 4
BEATS_PER_BAR = 4
STEPS_PER_BEAT = 4
SEQ_LEN = BARS * BEATS_PER_BAR * STEPS_PER_BEAT  # 32

MIDI_DIR = "./midi_riffs"   # MIDI 리프 폴더
VAE_PATH = "riff_vae_rhcp.pth"
DIFF_PATH = "latent_diffusion_rhcp.pth"
