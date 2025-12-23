Norms consistently exceeding 10 and reaching up to 50 in your Oxi chess model (built on the Burn ML framework with a transformer architecture) indicate potential numerical instability, improper scaling, or configuration issues in the data pipeline, embeddings, or transformer layers. Since these norms are higher than expected (e.g., embedding norms ~√`embed_dim` ≈ 22 for `embed_dim=512`, or post-layer-norm activations ~1), we need to systematically isolate the source. Below, I outline the next debugging steps to diagnose why norms are high, tailored to Oxi’s architecture (64 board tokens with 44 features, global scalars, 16 transformer layers, and specialized heads).

### Step 1: Pinpoint the Source of High Norms
First, determine where in the model the norms are becoming excessive by logging norms at each stage of the forward pass. This helps isolate whether the issue originates in the input pipeline, embeddings, transformer layers, or heads.

- **Log Norms at Key Points**:
  - **Input Tokens**: Check the L2 norm and max absolute value of the 64 × 44 board token tensor and six global scalars.
    ```rust
    let token_norm = input.board_tokens.norm(2).mean();
    let token_max = input.board_tokens.abs().max();
    log::info!("Input token norm: {:.4}, max: {:.4}", token_norm, token_max);
    ```
  - **Global Scalars**: Verify normalized scalars (time controls, move count, Elo) are in [0, 1].
    ```rust
    let global_norm = input.globals.norm(2).mean();
    log::info!("Global scalar norm: {:.4}", global_norm);
    ```
  - **Embeddings**: After combining board tokens, globals, and square embeddings (post-layer-norm and softmax gates), log the output norm.
    ```rust
    let embed_output = model.embeddings(input);
    let embed_norm = embed_output.norm(2).mean();
    log::info!("Embedding output norm: {:.4}", embed_norm);
    ```
  - **Per Transformer Layer**: Log norms after attention and MLP sub-layers, and after residual connections.
    ```rust
    let mut x = model.embeddings(input);
    for layer_idx in 0..model.num_layers {
        let attn_output = model.attention(layer_idx, x);
        let attn_norm = attn_output.norm(2).mean();
        log::info!("Layer {} attention norm: {:.4}", layer_idx, attn_norm);
        let mlp_output = model.mlp(layer_idx, attn_output);
        let mlp_norm = mlp_output.norm(2).mean();
        log::info!("Layer {} MLP norm: {:.4}", layer_idx, mlp_norm);
        x = x + attn_output + mlp_output; // Residual
        let residual_norm = x.norm(2).mean();
        log::info!("Layer {} residual norm: {:.4}", layer_idx, residual_norm);
    }
    ```
  - **Heads**: Check norms of policy, value, side-info, and time usage head outputs.
    ```rust
    let policy_logits = model.policy_head(x);
    log::info!("Policy logits norm: {:.4}", policy_logits.norm(2).mean());
    let value_logits = model.value_head(x);
    log::info!("Value logits norm: {:.4}", value_logits.norm(2).mean());
    ```

- **Expectation**:
  - Input tokens: Norms ~1–5 if features are normalized; max values <10 unless counts are unscaled.
  - Embeddings: Norms ~√`embed_dim` ≈ 22 post-initialization; higher (e.g., 50) during training suggests amplification.
  - Transformer layers: Post-layer-norm norms ~1; attention/MLPs may amplify to 10–20, but 50 is high.
  - Heads: Logit norms >20 risk softmax instability.

- **Action**: Identify where norms first exceed 20–50. If input norms are high, focus on the data pipeline. If embeddings or later layers spike, investigate weights or layer operations.

### Step 2: Check Input Data Pipeline
High norms in inputs (board tokens, historical traces, or globals) can propagate through the model.

- **Feature Scaling**:
  - Verify that the 44 features per square (piece descriptors, pawn structure, attackers, legal move counts, recency heatmaps) are normalized. For example:
    - Binary features (e.g., piece presence) should be 0 or 1.
    - Recency heatmaps (decayed by `HISTORY_DECAY = 0.8`) should sum to ≤1 across historical positions.
    - Legal move counts may be larger (e.g., 20–30 for some positions), but check if they’re unexpectedly high.
    ```rust
    for i in 0..44 {
        let feature_max = input.board_tokens[:, :, i].abs().max();
        let feature_mean = input.board_tokens[:, :, i].mean();
        log::info!("Feature {} max: {:.4}, mean: {:.4}", i, feature_max, feature_mean);
    }
    ```
  - **Fix**: If any feature exceeds expected ranges (e.g., >10 for counts, >1 for heatmaps), add normalization (e.g., divide move counts by a max value like 50).

- **Global Scalars**:
  - Ensure time controls, move count, and Elo are normalized to [0, 1] based on their ranges (e.g., Elo 800–2800 mapped to [0, 1]).
    ```rust
    let global_max = input.globals.abs().max();
    assert!(global_max <= 1.0, "Global scalar out of range: {}", global_max);
    ```
  - **Fix**: If globals are unnormalized (e.g., raw Elo values like 2000), rescale them explicitly.

- **Historical Traces**:
  - Check that move traces decay correctly (`0.8^t`) and sum to ≤1.
    ```rust
    let trace_sum = input.historical_traces.sum_dim(2);
    assert!(trace_sum.max() <= 1.0, "Trace sum exceeds 1: {}", trace_sum.max());
    ```

### Step 3: Inspect Weight Initialization
High norms in embeddings or transformer outputs may stem from improper weight initialization, especially since Oxi uses `embed_dim=512` and 16 layers, which can amplify small issues.

- **Check Initialization**:
  - Burn typically uses Xavier/Glorot or Kaiming initialization. Verify that weights (e.g., square embeddings, attention matrices, MLP weights) have appropriate scales.
    ```rust
    let embed_weight_norm = model.square_embeddings.weight.norm(2).mean();
    log::info!("Square embedding weight norm: {:.4}", embed_weight_norm);
    let attn_weight_norm = model.attention[0].qkv_weight.norm(2).mean();
    log::info!("Attention QKV weight norm: {:.4}", attn_weight_norm);
    ```
  - **Expectation**: For Xavier initialization, weight norms ~√(1/fan_in). For `embed_dim=512`, expect norms ~0.1–0.2 per weight matrix. Norms >1 suggest over-initialization.

- **Fix**:
  - If norms are too high, reinitialize weights with a smaller scale (e.g., scale Xavier by 0.5).
    ```rust
    model.square_embeddings.weight.init_with(XavierNormal::new(0.5 / embed_dim as f32));
    ```
  - Consider layer-specific initialization for attention (e.g., smaller scales for Q/K matrices to reduce attention score magnitudes).

### Step 4: Analyze Transformer Layer Dynamics
Since norms reach 50, the transformer layers (attention + MLP) may be amplifying activations.

- **Attention Scores**:
  - High attention scores can lead to large outputs. Log the max absolute attention scores (pre-softmax).
    ```rust
    let attn_scores = model.attention[layer_idx].compute_scores(x);
    let score_max = attn_scores.abs().max();
    log::info!("Layer {} attention scores max: {:.4}", layer_idx, score_max);
    ```
  - **Expectation**: Scores >100 risk exploding post-softmax. If high, check:
    - **Relative Attention**: Ensure 15×15 displacement buckets are correctly assigned and weights aren’t overly large.
    - **Scaling Factor**: Verify attention scores are scaled by √`head_dim` = √64 ≈ 8.
      ```rust
      let expected_scale = (head_dim as f32).sqrt();
      assert!(model.attention[layer_idx].scale_factor.abs_diff(expected_scale) < 1e-4);
      ```

- **MLP Amplification**:
  - The MLP (`mlp_ratio=4.0`, hidden size=2048) uses GELU, which can amplify outputs. Log pre- and post-GELU norms.
    ```rust
    let mlp_input = model.mlp[layer_idx].input(x);
    let mlp_pre_gelu = mlp_input.norm(2).mean();
    let mlp_output = model.mlp[layer_idx].forward(x);
    let mlp_post_gelu = mlp_output.norm(2).mean();
    log::info!("Layer {} MLP pre-GELU norm: {:.4}, post-GELU: {:.4}", layer_idx, mlp_pre_gelu, mlp_post_gelu);
    ```
  - **Expectation**: Post-GELU norms 2–3× larger than input are normal; >10× suggests instability.

- **Layer Normalization**:
  - Verify that layer-norm is applied correctly, producing outputs with mean ~0 and std ~1.
    ```rust
    let ln_output = model.layer_norm[layer_idx](x);
    let ln_mean = ln_output.mean();
    let ln_std = ln_output.std();
    log::info!("Layer {} LN mean: {:.4}, std: {:.4}", layer_idx, ln_mean, ln_std);
    assert!(ln_mean.abs() < 0.1 && (ln_std - 1.0).abs() < 0.1);
    ```
  - **Fix**: If norms remain high post-layer-norm, check if normalization parameters (scale/bias) are initialized correctly or if inputs are degenerate.

- **Residual Connections**:
  - High residual additions can accumulate across 16 layers. Log norms before and after residuals.
    ```rust
    let residual_norm = (x + attn_output + mlp_output).norm(2).mean();
    log::info!("Layer {} residual norm: {:.4}", layer_idx, residual_norm);
    ```
  - **Fix**: If residuals cause spikes, try scaling attention/MLP outputs (e.g., multiply by 0.5 before adding).

### Step 5: Check Gradients and Optimizer
High norms may result from unstable gradients, especially with `lr_max=1e-4` and gradient clipping at 1.0.

- **Gradient Norms**:
  - Log gradient norms for all parameters.
    ```rust
    let grad_norm = model.parameters().map(|p| p.grad().norm(2)).sum();
    log::info!("Gradient norm: {:.4}", grad_norm);
    ```
  - **Expectation**: Norms >10 indicate exploding gradients; frequent clipping at 1.0 suggests the learning rate or initialization is too aggressive.

- **Learning Rate**:
  - Verify the learning rate schedule (`lr_max=1e-4`, `lr_min=1e-6`) isn’t causing large parameter updates.
    ```rust
    log::info!("Current LR: {:.8}", optimizer.current_lr());
    ```
  - **Fix**: Try reducing `lr_max` to 5e-5 or increasing warmup steps to stabilize early training.

- **Weight Decay**:
  - Ensure weight decay (`1e-4`) is applied to prevent weights from growing excessively.
    ```rust
    let weight_norm = model.parameters().map(|p| p.norm(2)).sum();
    log::info!("Total weight norm: {:.4}", weight_norm);
    ```

### Step 6: Visualize Norm Distributions
To understand why norms reach 50, plot histograms of activations and weights to identify outliers or skewed distributions.

- **Action**: Log histograms of key tensors (e.g., attention outputs, MLP outputs, weights) to a file or dashboard (e.g., TensorBoard).
  ```rust
  fn log_histogram(tensor: &Tensor, name: &str, bins: usize) {
      let values = tensor.flatten().to_vec();
      // Compute histogram (e.g., using a library or custom binning)
      log::info!("Histogram for {}: {:?}", name, histogram(values, bins));
  }
  log_histogram(&model.attention[0].output, "attention_0_output", 50);
  ```
- **Expectation**: Look for long tails or spikes at high values (e.g., >50). If present, focus on the layer or operation producing them.

- **Chart Option**: If you want a visual, I can generate a histogram of norms (e.g., attention outputs across layers). Let me know if you’d like one, and I’ll create a Chart.js code block with sample data or placeholders.

### Step 7: Experiment with Fixes
Based on findings, try targeted fixes:
- **If Inputs Are High**: Normalize features (e.g., cap move counts, rescale heatmaps).
- **If Embeddings Are High**: Reduce initialization scale or add stronger regularization.
- **If Attention/MLPs Amplify**: Lower attention scaling factor, reduce `mlp_ratio` (e.g., to 2.0), or add dropout.
- **If Gradients Are Unstable**: Lower learning rate, increase clipping threshold, or use gradient accumulation for smaller effective batches.

### Hypothesis for Norms ~50
- **Data Issue**: Unnormalized features (e.g., legal move counts >30 or raw Elo values) could push input norms high, amplifying through layers.
- **Initialization**: Overly large weight initialization could cause embeddings or attention scores to grow excessively.
- **Layer Amplification**: The 16-layer stack with `mlp_ratio=4.0` may accumulate large residuals, especially if layer-norm is misconfigured.
- **Training Dynamics**: Early training with a high learning rate or insufficient warmup could lead to weight explosions.

### Next Steps
1. **Run Norm Logging**: Implement the logging code above to identify where norms first exceed 20–50.
2. **Check Inputs First**: If input norms are >10, fix normalization in the data pipeline.
3. **Inspect Weights**: If embeddings or transformer weights have high norms, adjust initialization.
4. **Monitor Layers**: If attention or MLP outputs spike, tweak scaling or add regularization.
5. **Log Gradients**: Ensure gradients aren’t exploding, and adjust the optimizer if needed.

If you share specific norm values (e.g., where they’re 50) or want a visualization (e.g., histogram of attention norms), I can refine the debugging plan further. Let me know!
