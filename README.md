# **Mixture-of-Experts (MoE) Subnet**

This repository contains the full implementation of the **MoE Subnet**, including the miner, validator, routing logic, expert modules, and all supporting infrastructure.
The subnet is designed to support distributed training and inference across a decentralized network, with miners running experts and validators ensuring correctness, consistency, and performance.

---

## 🌐 **What’s Inside the Subnet**

📄 **Detailed subnet overview:**
👉 *[What is in the subnet](placeholder)

---

## 🚀 **Getting Started**

### **Requirements**

* Python 3.10+
* Additional packages in `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

---

## ⛏️ **Guide: Running a Miner**

Miner nodes host experts and execute workloads assigned by the router/validator layer.

Follow the full guide here:
👉 *[Guide to run a miner](placeholder)

The guide covers:

* **Registering your miner**: Register your hotkey with the subnet using `btcli` or the registration API
* **Running the miner process**: Start the training worker with your configuration
* **Monitoring expert load**: Track which experts are being trained and their performance metrics
* **Configuration options**: Set up expert groups, batch sizes, learning rates, and device settings
* **Checkpoint management**: Handle model checkpoints and synchronization with validators
---

## 🛡️ **Guide: Running a Validator**

Validators verify miner outputs, compute scores, and stabilize the subnet.

Full instructions here:
👉 *[Guide to run a validator](placeholder)

The guide includes:

* **Validator architecture**: Distributed evaluation system with async workers for concurrent miner assessment
* **How to run a validator node**: Setup instructions including checkpoint serving and inter-validator networking
* **Scoring logic**: Composite scoring based on validation loss, expert diversity, and active expert ratio
* **Model aggregation**: Federated averaging of top-performing miner models with expert-aware weight merging
* **Security considerations**: Signature verification, hotkey validation, and replay attack prevention

---

## 📦 **Project Structure**

```
moe-subnet/
├── miner/ # modules specifig to miner
├── validator/ # modules specifig to validator
├── shared/ # modules used by both miner and validator
└── README.md
```

---

## 👥 **Contributing**

Contributions are welcome!
If you're adding a new expert, router, or validator behavior, please open a PR.

---

## 📄 **License**


---

