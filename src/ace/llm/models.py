from enum import StrEnum


class ModelsEnum(StrEnum):
    """Currently supported models."""

    GPT_4O_2024_05_13 = "gpt-4o-2024-05-13"
    GPT_4O_MINI_2024_07_18 = "gpt-4o-mini-2024-07-18"
    GPT_4_0125_PREVIEW = "gpt-4-0125-preview"
    GPT_3_5_TURBO_0125 = "gpt-3.5-turbo-0125"
    GPT_4_TURBO_2024_04_09 = "gpt-4-turbo-2024-04-09"
    CLAUDE_3_OPUS_20240229 = "claude-3-opus-20240229"
    CLAUDE_3_SONNET_20240229 = "claude-3-sonnet-20240229"
    CLAUDE_3_5_SONNET_20240620 = "claude-3-5-sonnet-20240620"
    CLAUDE_3_5_SONNET_20241022 = "claude-3-5-sonnet-20241022"
    CLAUDE_3_7_SONNET_20250219 = "claude-3-7-sonnet-20250219"
    CLAUDE_3_7_SONNET_20250219_THINKING_16000 = "claude-3-7-sonnet-20250219-thinking-16000"
    CLAUDE_3_HAIKU_20240307 = "claude-3-haiku-20240307"
    COMMAND_R_PLUS = "command-r-plus"
    COMMAND_R = "command-r"
    MISTRALAI_MIXTRAL_8X7B_INSTRUCT_V0_1 = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    LLAMA_3 = "meta-llama/Llama-3-70b-chat-hf"
    GEMINI_1_5_PRO_002 = "gemini-1.5-pro-002"
    GEMINI_1_5_PRO_001 = "gemini-1.5-pro-001"
    GEMINI_1_5_FLASH_002 = "gemini-1.5-flash-002"
    GEMINI_1_5_FLASH_001 = "gemini-1.5-flash-001"
    GEMINI_2_0_FLASH_EXP = "gemini-2.0-flash-exp"
    GEMINI_2_0_FLASH_001 = "gemini-2.0-flash-001"
    GEMINI_2_5_FLASH_PREVIEW_04_17 = "gemini-2.5-flash-preview-04-17"
    GEMINI_2_5_PRO_PREVIEW_05_06 = "gemini-2.5-pro-preview-05-06"
    LOCAL = "local"
    VLLM_PARSED = "vllm_parsed"


MODEL_PROVIDERS = {
    ModelsEnum.GPT_4O_2024_05_13: "openai",
    ModelsEnum.GPT_4O_MINI_2024_07_18: "openai",
    ModelsEnum.GPT_3_5_TURBO_0125: "openai",
    ModelsEnum.GPT_4_0125_PREVIEW: "openai",
    ModelsEnum.GPT_4_TURBO_2024_04_09: "openai",
    ModelsEnum.CLAUDE_3_OPUS_20240229: "anthropic",
    ModelsEnum.CLAUDE_3_SONNET_20240229: "anthropic",
    ModelsEnum.CLAUDE_3_5_SONNET_20240620: "anthropic",
    ModelsEnum.CLAUDE_3_5_SONNET_20241022: "anthropic",
    ModelsEnum.CLAUDE_3_7_SONNET_20250219: "anthropic",
    ModelsEnum.CLAUDE_3_7_SONNET_20250219_THINKING_16000: "anthropic",
    ModelsEnum.CLAUDE_3_HAIKU_20240307: "anthropic",
    ModelsEnum.COMMAND_R_PLUS: "cohere",
    ModelsEnum.COMMAND_R: "cohere",
    ModelsEnum.MISTRALAI_MIXTRAL_8X7B_INSTRUCT_V0_1: "together",
    ModelsEnum.LLAMA_3: "together-prompting",
    ModelsEnum.GEMINI_1_5_PRO_001: "google",
    ModelsEnum.GEMINI_1_5_PRO_002: "google",
    ModelsEnum.GEMINI_1_5_FLASH_001: "google",
    ModelsEnum.GEMINI_1_5_FLASH_002: "google",
    ModelsEnum.GEMINI_2_0_FLASH_EXP: "google",
    ModelsEnum.GEMINI_2_0_FLASH_001: "google",
    ModelsEnum.GEMINI_2_5_FLASH_PREVIEW_04_17: "google",
    ModelsEnum.GEMINI_2_5_PRO_PREVIEW_05_06: "google",
    ModelsEnum.LOCAL: "local",
    ModelsEnum.VLLM_PARSED: "vllm_parsed",
}


MODEL_NAMES = {
    ModelsEnum.GPT_4O_2024_05_13: "GPT-4o (May 2024)",
    ModelsEnum.GPT_4O_MINI_2024_07_18: "GPT-4o Mini (July 2024)",
    ModelsEnum.GPT_4_0125_PREVIEW: "GPT-4 Turbo Preview 0125",
    ModelsEnum.GPT_3_5_TURBO_0125: "GPT-3.5 Turbo 0125",
    ModelsEnum.GPT_4_TURBO_2024_04_09: "GPT-4 Turbo",
    ModelsEnum.CLAUDE_3_OPUS_20240229: "Claude 3 Opus",
    ModelsEnum.CLAUDE_3_SONNET_20240229: "Claude 3 Sonnet",
    ModelsEnum.CLAUDE_3_5_SONNET_20240620: "Claude 3.5 Sonnet",
    ModelsEnum.CLAUDE_3_5_SONNET_20241022: "Claude 3.5 Sonnet October",
    ModelsEnum.CLAUDE_3_7_SONNET_20250219: "Claude 3.7 Sonnet",
    ModelsEnum.CLAUDE_3_7_SONNET_20250219_THINKING_16000: "Claude 3.7 Sonnet (16k)",
    ModelsEnum.CLAUDE_3_HAIKU_20240307: "Claude 3 Haiku",
    ModelsEnum.COMMAND_R_PLUS: "Command R Plus",
    ModelsEnum.COMMAND_R: "Command R",
    ModelsEnum.MISTRALAI_MIXTRAL_8X7B_INSTRUCT_V0_1: "Mixtral 8x7B Instruct v0.1",
    ModelsEnum.LLAMA_3: "Llama 3",
    ModelsEnum.GEMINI_1_5_PRO_001: "Gemini 1.5 Pro 001",
    ModelsEnum.GEMINI_1_5_PRO_002: "Gemini 1.5 Pro 002",
    ModelsEnum.GEMINI_1_5_FLASH_001: "Gemini 1.5 Flash 001",
    ModelsEnum.GEMINI_1_5_FLASH_002: "Gemini 1.5 Flash 002",
    ModelsEnum.GEMINI_2_0_FLASH_EXP: "Gemini 2.0 Flash Exp",
    ModelsEnum.GEMINI_2_0_FLASH_001: "Gemini 2.0 Flash 001",
    ModelsEnum.GEMINI_2_5_FLASH_PREVIEW_04_17: "Gemini 2.5 Flash Preview 04/17",
    ModelsEnum.GEMINI_2_5_PRO_PREVIEW_05_06: "Gemini 2.5 Pro Preview 05/06",
    ModelsEnum.LOCAL: "Local model",
    ModelsEnum.VLLM_PARSED: "VLLM parsed model",
}
