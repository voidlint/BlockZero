# MoE Subnet Documentation

Complete documentation for the Mixture-of-Experts decentralized subnet.

## 📚 Documentation Structure

### Architecture
Deep dives into system design and technical components:

- **[MoE Architecture](architecture/moe_architecture.md)** - Complete overview of the distributed MoE system
  - Model structure (Qwen3-VL-MoE 30B)
  - Expert group distribution
  - Training cycle phases
  - Memory optimization strategies

- **[Vision Pipeline](architecture/vision_pipeline.md)** - Multimodal processing details
  - Vision encoder architecture
  - Image preprocessing pipeline
  - Vision-specific metrics (VQA, captioning)
  - Anti-cheat mechanisms

- **[ESFT Overview](architecture/esft_overview.md)** - Expert selection methodology
  - Research findings & benchmarks
  - Profiling methods (gate-based vs token-based)
  - Distributed ESFT workflow
  - Security considerations

### Guides
Step-by-step setup and operation guides:

- **[Miner Setup](guides/miner_setup.md)** - Complete miner onboarding
  - Hardware requirements (A6000/A100)
  - Installation & configuration
  - Running & monitoring
  - Troubleshooting common issues
  - Performance optimization tips

- **[Validator Setup](guides/validator_setup.md)** - Validator node setup
  - System requirements
  - Configuration & deployment
  - Evaluation & scoring
  - Inter-validator consensus

- **[Subnet Owner Guide](guides/subnet_owner_guide.md)** - Network administration
  - Expert assignment management
  - ESFT execution & publishing
  - Phase coordination
  - Network monitoring

### API Reference
Detailed function and API documentation:

- **[ESFT API](api/esft_api.md)** - Expert selection functions
  - `ESFTExpertSelector` class
  - `run_esft_expert_selection()`
  - `load_expert_assignment()`
  - Configuration options

- **[Metrics API](api/metrics_api.md)** - Evaluation metrics
  - Expert-specific metrics (math, agentic, planning, vision)
  - Composite scoring
  - Score aggregation
  - Custom metric implementation

- **[Profiler API](api/profiler_api.md)** - Expert profiling tools
  - `ExpertProfiler` usage
  - Skill matrix generation
  - Specialization analysis
  - Routing pattern visualization

## 🔒 Security Documentation

- **[Centralized Expert Assignment](CENTRALIZED_EXPERT_ASSIGNMENT.md)** - Assignment system overview
- **[ESFT Security](ESFT_SECURITY.md)** - Preventing miner bypass attacks

## 🚀 Quick Start

**New to the subnet?** Start here:

1. Read [MoE Architecture](architecture/moe_architecture.md) for system overview
2. Follow [Miner Setup Guide](guides/miner_setup.md) to start training
3. Review [ESFT Security](ESFT_SECURITY.md) for security best practices

**Running a validator?**

1. See [Validator Setup](guides/validator_setup.md)
2. Review [Metrics API](api/metrics_api.md) for evaluation details

**Managing the network?**

1. Follow [Subnet Owner Guide](guides/subnet_owner_guide.md)
2. Use [ESFT API](api/esft_api.md) for expert selection

## 📊 Key Specifications

| Parameter | Value |
|-----------|-------|
| Model | Qwen3-VL-MoE (30B params) |
| Expert Groups | 4 (Math, Agentic, Planning, Vision) |
| Experts per Layer | 4 |
| Total Layers | 48 |
| Active Experts | 2 per token (top-2 routing) |
| Training Cycle | 600 blocks (~1 hour) |
| Min GPU | A6000 48GB (with quantization) |

## 🤝 Contributing

Found an issue or want to improve the docs?

1. Fork the repository
2. Make your changes
3. Submit a pull request

## 📞 Support

- **GitHub Issues:** [Report bugs](https://github.com/CognitoBlocks/BlockZero/issues)
- **Discord:** [Join community](https://discord.gg/bittensor)
- **Documentation:** You're reading it! 📖

## 📝 Documentation Status

| Section | Status | Last Updated |
|---------|--------|--------------|
| Architecture | ✅ Complete | 2026-01-14 |
| Miner Guide | ✅ Complete | 2026-01-14 |
| Validator Guide | 🚧 In Progress | 2026-01-14 |
| Owner Guide | 🚧 In Progress | 2026-01-14 |
| API Reference | 🚧 In Progress | 2026-01-14 |
| Security Docs | ✅ Complete | 2026-01-14 |

## 🔄 Recent Updates

- **2026-01-14:** Added centralized expert assignment system
- **2026-01-14:** Migrated from Qwen3-Next to Qwen3-VL-MoE
- **2026-01-14:** Added ESFT security documentation
- **2026-01-14:** Complete architecture documentation

