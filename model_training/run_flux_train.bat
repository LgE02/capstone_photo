@echo off
echo ============================================================
echo FLUX LoRA 학습 (2단계)
echo ============================================================

echo.
echo [1/2] 텍스트 임베딩 사전 계산...
echo.
python model_training/compute_embeddings.py

if %errorlevel% neq 0 (
    echo 임베딩 계산 실패!
    pause
    exit /b 1
)

echo.
echo [2/2] LoRA 학습 시작...
echo.
accelerate launch model_training/train_dreambooth_lora_flux_miniature.py ^
  --pretrained_model_name_or_path="black-forest-labs/FLUX.1-schnell" ^
  --data_df_path="model_training/embeddings.parquet" ^
  --output_dir="lora_output/flux_lora" ^
  --logging_dir="lora_output/flux_lora/logs" ^
  --mixed_precision="bf16" ^
  --rank=4 ^
  --resolution=512 ^
  --train_batch_size=1 ^
  --gradient_accumulation_steps=4 ^
  --learning_rate=1e-4 ^
  --lr_scheduler="cosine" ^
  --lr_warmup_steps=50 ^
  --max_train_steps=700 ^
  --checkpointing_steps=100 ^
  --use_8bit_adam ^
  --cache_latents ^
  --seed=42 ^
  --center_crop ^
  --random_flip ^
  --report_to="none"

echo.
echo ============================================================
echo 학습 완료! 결과: lora_output/flux_lora/
echo ============================================================
pause
