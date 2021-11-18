# Vitis-AI WAA (Whole Application Accleration) for Edge

**Framework版本**

`Vitis AI V1.4.1`

`Vitis AI Library V1.4.1`

`vitis 2020.2`



**DPU配置 for xmodel**

`Dual B4096`

`RAM Usage LOW`

`DSP48 Usage HIGH`


- **主要脚本程序：**
`finetune.sh`

- **基本流程：**

	```mermaid
  graph LR
      B([float模型])-->C[量化]-->D([int8模型])-->A[剪枝]-->E([剪枝后的int8模型])-->F[导出模型权重]-->G[读取模型权重]
  
  ```
    **第一步**：

    量化裁减网络并导出模型参数。
    ```shell
	  python quant_fast_finetune.py --quant_mode calib --fast_finetune
    ```
    核心代码：
    ```python
        quantizer = torch_quantizer(
            quant_mode, model, (input), device=device)
        quant_model = quantizer.quant_model
  
        quantizer.fast_finetune(evaluate, (quant_model, ft_loader, register_buffers))
        quantizer.export_quant_config()
    ```
    **第二步**：

    读取参数并导出`.xmodel`
    ```shell
    python quant_fast_finetune.py  --quant_mode test --subset_len 1 --batch_size=8 --fast_finetune --deploy
    ```
    核心代码：
    ```python
        quantizer.load_ft_param()
        quant_model = quantizer.quant_model
        quantizer.export_xmodel(deploy_check=False)
    ```
  
    常用`Graph`结构可视化:
    ```python
    for node in graph._dev_graph.nodes:
    	print(node.name, node.op.type)p
    
    ```
  

