---
name: deploy-bot
description: Deploy a trained Oxi model to production - downloads from remote, uploads to GCS, updates bot deployment, and pushes code changes
---

# Deploy Bot Skill

## When to Use

Use this skill when:
- A training run has completed on the remote GPU server
- You need to deploy a new model version to production
- The user says "deploy the model" or "copy model to prod"

## Prerequisites

- SSH access to remote server via `REMOTE_IP` env var (use `mise exec -- printenv REMOTE_IP`)
- GCS credentials at `../secrets/chessbook-svc-key` (relative to oxi directory)
- Git repos: `oxi` (this repo) and `server` (at `../server`)

## Deployment Steps

### Step 1: Verify Remote Model

Check the model on the remote server:

```bash
ssh ubuntu@$(mise exec -- printenv REMOTE_IP) 'bash -l -c "ls -la /home/ubuntu/oxi/model/"'
```

Expected files:
- `model.mpk` - The trained model weights
- `params.json` - Training parameters (architecture config)
- `optimizer_*.mpk` - Optimizer states (not needed for inference)
- `gradnorm_state.bin` - GradNorm state (not needed for inference)

Verify the model is recent (check timestamp is within expected training window).

### Step 2: Get Training Parameters

Read the params.json to understand model architecture:

```bash
ssh ubuntu@$(mise exec -- printenv REMOTE_IP) 'bash -l -c "cat /home/ubuntu/oxi/model/params.json"'
```

Key parameters to note:
- `embed_dim` - Embedding dimension (e.g., 448)
- `num_layers` - Number of transformer layers (e.g., 18)
- `num_heads` - Attention heads (e.g., 16)
- `smolgen_*` - Smolgen configuration

If params.json doesn't exist (older training runs), check train.log header:

```bash
ssh ubuntu@$(mise exec -- printenv REMOTE_IP) 'bash -l -c "head -50 /home/ubuntu/oxi/train.log"'
```

### Step 3: Create Model Directory Name

Create a descriptive name based on architecture:

Format: `model-{embed_dim}-{num_layers}L`

Examples:
- `model-320-24L` (320 embed dim, 24 layers)
- `model-448-18L` (448 embed dim, 18 layers)

### Step 4: Download Model from Remote

Download model.mpk and params.json to local:

```bash
mkdir -p /tmp/oxi-model-deploy
scp ubuntu@$(mise exec -- printenv REMOTE_IP):/home/ubuntu/oxi/model/model.mpk /tmp/oxi-model-deploy/
scp ubuntu@$(mise exec -- printenv REMOTE_IP):/home/ubuntu/oxi/model/params.json /tmp/oxi-model-deploy/
```

### Step 5: Upload to Google Cloud Storage

Authenticate with GCS:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="../secrets/chessbook-svc-key"
```

Upload to GCS bucket with the model directory name:

```bash
MODEL_NAME="model-{embed_dim}-{num_layers}L"  # Replace with actual values
gsutil -o "Credentials:gs_service_key_file=$GOOGLE_APPLICATION_CREDENTIALS" cp /tmp/oxi-model-deploy/model.mpk gs://chessbook-models/oxi/${MODEL_NAME}/
gsutil -o "Credentials:gs_service_key_file=$GOOGLE_APPLICATION_CREDENTIALS" cp /tmp/oxi-model-deploy/params.json gs://chessbook-models/oxi/${MODEL_NAME}/
```

Verify upload:

```bash
gsutil -o "Credentials:gs_service_key_file=$GOOGLE_APPLICATION_CREDENTIALS" ls -la gs://chessbook-models/oxi/${MODEL_NAME}/
```

### Step 6: Update Bot Deployment Config

Edit `../server/bot/deploy.yml` to use the new model:

Find the line:
```yaml
gcloud storage cp -r gs://chessbook-models/oxi/model-XXX-YYL/* /models/
```

Update to:
```yaml
gcloud storage cp -r gs://chessbook-models/oxi/${MODEL_NAME}/* /models/
```

### Step 7: Push Code Changes

Push changes to both repositories:

**Oxi repo** (this repo):
```bash
cd /Users/marcusbuffett/projects/chessbook/oxi
git add -A
git commit -m "Add params.json saving to training checkpoints"
git push origin main
```

**Server repo** (bot deployment):
```bash
cd /Users/marcusbuffett/projects/chessbook/server
git add bot/deploy.yml
git commit -m "Update bot to use ${MODEL_NAME}"
git push origin main
```

### Step 8: Trigger Deployment

The Cloud Build trigger will automatically deploy when changes are pushed to `bot/**` in the server repo.

Monitor deployment:

```bash
gcloud builds list --region=us-central1 --project=chessbook-404210 --limit=5
```

Check pod status:

```bash
kubectl get pods -l app=chess-bot
kubectl logs -l app=chess-bot --tail=50
```

## Verification

Test the bot endpoint:

```bash
kubectl port-forward svc/chess-bot 8402:8402 &
curl -X POST http://localhost:8402/predict \
  -H "Content-Type: application/json" \
  -d '{
    "history": [],
    "whiteTimeMs": 300000,
    "blackTimeMs": 300000,
    "timeControl": {"initialTimeMs": 300000, "incrementMs": 0},
    "botElo": 1500,
    "topN": 5
  }'
```

## Cleanup

Remove temporary files:

```bash
rm -rf /tmp/oxi-model-deploy
```

## Troubleshooting

### Model download fails
- Check SSH connectivity: `ssh ubuntu@$(mise exec -- printenv REMOTE_IP) 'echo ok'`
- Verify model exists on remote

### GCS upload fails
- Verify credentials file exists at `../secrets/chessbook-svc-key`
- Check bucket permissions

### Bot fails to start
- Check init container logs: `kubectl logs -l app=chess-bot -c download-model`
- Verify model path in GCS matches deploy.yml

### Model inference errors
- Ensure params.json matches the model architecture
- Check RUST_LOG for detailed error messages
