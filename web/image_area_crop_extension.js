import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";
import { api } from "../../scripts/api.js";

// 加载CSS样式
const link = document.createElement('link');
link.rel = 'stylesheet';
link.type = 'text/css';
link.href = '/extensions/ComfyUI-ImageAreaCrop/web/image_area_crop.css';
document.head.appendChild(link);

class ImageAreaCropExtension {
    constructor() {
        this.node = null;
        this.canvas = null;
        this.ctx = null;
        this.isDragging = false;
        this.startX = 0;
        this.startY = 0;
        this.currentX = 0;
        this.currentY = 0;
        this.imageData = null;
        this.scale = 1;
        this.offsetX = 0;
        this.offsetY = 0;
        this.hasSelection = false;
    }

    bindNode(node) {
        this.node = node;
    }

    createVisualInterface() {
        if (!this.node) return;

        // 创建主容器
        const container = document.createElement('div');
        container.className = 'image-crop-container';
        
        // 创建画布
        this.canvas = document.createElement('canvas');
        this.canvas.className = 'image-crop-canvas';
        this.canvas.width = 400;
        this.canvas.height = 300;
        this.ctx = this.canvas.getContext('2d');
        
        // 创建控制按钮
        const controls = document.createElement('div');
        controls.className = 'image-crop-controls';
        
        const resetBtn = document.createElement('button');
        resetBtn.textContent = '清除裁切信息';
        resetBtn.className = 'visual-crop-btn';
        resetBtn.onclick = () => this.resetSelection();
        
        const applyBtn = document.createElement('button');
        applyBtn.textContent = '应用选择';
        applyBtn.className = 'visual-crop-btn primary';
        applyBtn.onclick = () => this.applySelection();

        const sizeInfo = document.createElement('div');
        sizeInfo.className = 'visual-crop-info';
        sizeInfo.textContent = '精准尺寸控制';

        const xInput = document.createElement('input');
        xInput.type = 'number';
        xInput.min = '0';
        xInput.placeholder = 'X';
        xInput.style.width = '70px';
        xInput.onchange = () => {
            const x = Math.max(0, parseInt(xInput.value || '0'));
            this.startX = this.offsetX + x * this.scale;
            this.currentX = Math.max(this.startX, this.currentX);
            this.drawCropArea();
            this.updateNodeParameters();
        };

        const yInput = document.createElement('input');
        yInput.type = 'number';
        yInput.min = '0';
        yInput.placeholder = 'Y';
        yInput.style.width = '70px';
        yInput.onchange = () => {
            const y = Math.max(0, parseInt(yInput.value || '0'));
            this.startY = this.offsetY + y * this.scale;
            this.currentY = Math.max(this.startY, this.currentY);
            this.drawCropArea();
            this.updateNodeParameters();
        };

        const wInput = document.createElement('input');
        wInput.type = 'number';
        wInput.min = '1';
        wInput.placeholder = '宽度';
        wInput.style.width = '80px';
        wInput.onchange = () => {
            const width = Math.max(1, parseInt(wInput.value || '0'));
            this.currentX = this.startX + width * this.scale;
            this.drawCropArea();
            this.updateNodeParameters();
        };

        const hInput = document.createElement('input');
        hInput.type = 'number';
        hInput.min = '1';
        hInput.placeholder = '高度';
        hInput.style.width = '80px';
        hInput.onchange = () => {
            const height = Math.max(1, parseInt(hInput.value || '0'));
            this.currentY = this.startY + height * this.scale;
            this.drawCropArea();
            this.updateNodeParameters();
        };

        controls.appendChild(resetBtn);
        controls.appendChild(applyBtn);
        controls.appendChild(xInput);
        controls.appendChild(yInput);
        controls.appendChild(wInput);
        controls.appendChild(hInput);
        controls.appendChild(sizeInfo);
        
        container.appendChild(this.canvas);
        container.appendChild(controls);
        
        // 绑定鼠标事件
        this.wInputRef = wInput;
        this.hInputRef = hInput;
        this.xInputRef = xInput;
        this.yInputRef = yInput;
        this.bindCanvasEvents();
        
        return container;
    }

    bindCanvasEvents() {
        if (!this.canvas) return;

        this.canvas.addEventListener('mousedown', (e) => {
            this.isDragging = true;
            const rect = this.canvas.getBoundingClientRect();
            const sx = this.canvas.width / rect.width;
            const sy = this.canvas.height / rect.height;
            this.startX = (e.clientX - rect.left) * sx;
            this.startY = (e.clientY - rect.top) * sy;
            this.currentX = this.startX;
            this.currentY = this.startY;
        });

        this.canvas.addEventListener('mousemove', (e) => {
            if (!this.isDragging) return;
            const rect = this.canvas.getBoundingClientRect();
            const sx = this.canvas.width / rect.width;
            const sy = this.canvas.height / rect.height;
            this.currentX = (e.clientX - rect.left) * sx;
            this.currentY = (e.clientY - rect.top) * sy;
            this.drawCropArea();
        });

        this.canvas.addEventListener('mouseup', () => {
            this.isDragging = false;
            const x = Math.min(this.startX, this.currentX);
            const y = Math.min(this.startY, this.currentY);
            const width = Math.abs(this.currentX - this.startX);
            const height = Math.abs(this.currentY - this.startY);
            const imgX = Math.round((x - this.offsetX) / this.scale);
            const imgY = Math.round((y - this.offsetY) / this.scale);
            const imgW = Math.round(width / this.scale);
            const imgH = Math.round(height / this.scale);
            if (this.wInputRef) this.wInputRef.value = Math.max(1, imgW);
            if (this.hInputRef) this.hInputRef.value = Math.max(1, imgH);
            if (this.xInputRef) this.xInputRef.value = Math.max(0, imgX);
            if (this.yInputRef) this.yInputRef.value = Math.max(0, imgY);
            this.hasSelection = imgW > 0 && imgH > 0;
            this.updateNodeParameters();
        });

        this.canvas.addEventListener('mouseleave', () => {
            this.isDragging = false;
        });
    }

    loadImage(imageUrl) {
        const img = new Image();
        img.onload = () => {
            this.imageData = img;
            this.fitImageToCanvas();
            this.drawCropArea();
        };
        img.src = imageUrl;
    }

    fitImageToCanvas() {
        if (!this.imageData || !this.canvas) return;

        const canvasWidth = this.canvas.width;
        const canvasHeight = this.canvas.height;
        const imgWidth = this.imageData.width;
        const imgHeight = this.imageData.height;

        // 计算缩放比例以适应画布
        const scaleX = canvasWidth / imgWidth;
        const scaleY = canvasHeight / imgHeight;
        this.scale = Math.min(scaleX, scaleY);

        // 计算居中偏移
        this.offsetX = (canvasWidth - imgWidth * this.scale) / 2;
        this.offsetY = (canvasHeight - imgHeight * this.scale) / 2;
    }

    drawCropArea() {
        if (!this.ctx || !this.canvas) return;

        // 清空画布
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // 绘制图像
        if (this.imageData) {
            this.ctx.drawImage(
                this.imageData,
                this.offsetX,
                this.offsetY,
                this.imageData.width * this.scale,
                this.imageData.height * this.scale
            );
        }

        // 绘制选择区域
        if (this.isDragging || (this.startX !== this.currentX && this.startY !== this.currentY)) {
            const x = Math.min(this.startX, this.currentX);
            const y = Math.min(this.startY, this.currentY);
            const width = Math.abs(this.currentX - this.startX);
            const height = Math.abs(this.currentY - this.startY);

            // 绘制半透明遮罩
            this.ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
            this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
            
            // 清除选择区域
            this.ctx.clearRect(x, y, width, height);
            
            // 重新绘制选择区域内的图像
            if (this.imageData) {
                this.ctx.drawImage(
                    this.imageData,
                    this.offsetX,
                    this.offsetY,
                    this.imageData.width * this.scale,
                    this.imageData.height * this.scale
                );
            }

            // 绘制选择框边框
            this.ctx.strokeStyle = '#00ff00';
            this.ctx.lineWidth = 2;
            this.ctx.strokeRect(x, y, width, height);

        const imgW = Math.round(width / this.scale);
        const imgH = Math.round(height / this.scale);
        this.ctx.fillStyle = '#00ff00';
        this.ctx.font = '12px Arial';
        this.ctx.fillText(`${imgW}x${imgH}`, x, Math.max(10, y - 5));
        }
    }

    updateNodeParameters() {
        if (!this.node || !this.imageData) return;

        const x = Math.min(this.startX, this.currentX);
        const y = Math.min(this.startY, this.currentY);
        const width = Math.abs(this.currentX - this.startX);
        const height = Math.abs(this.currentY - this.startY);

        // 转换画布坐标到图像坐标
        let imgX = Math.round((x - this.offsetX) / this.scale);
        let imgY = Math.round((y - this.offsetY) / this.scale);
        let imgWidth = Math.round(width / this.scale);
        let imgHeight = Math.round(height / this.scale);

        // 边界约束
        imgX = Math.max(0, Math.min(imgX, this.imageData.width - 1));
        imgY = Math.max(0, Math.min(imgY, this.imageData.height - 1));
        imgWidth = Math.max(1, Math.min(imgWidth, this.imageData.width - imgX));
        imgHeight = Math.max(1, Math.min(imgHeight, this.imageData.height - imgY));

        // 更新节点参数
        const xWidget = this.node.widgets.find(w => w.name === 'x');
        const yWidget = this.node.widgets.find(w => w.name === 'y');
        const widthWidget = this.node.widgets.find(w => w.name === 'width');
        const heightWidget = this.node.widgets.find(w => w.name === 'height');

        if (xWidget) xWidget.value = imgX;
        if (yWidget) yWidget.value = imgY;
        if (widthWidget) widthWidget.value = imgWidth;
        if (heightWidget) heightWidget.value = imgHeight;
    }

    resetSelection() {
        this.startX = 0;
        this.startY = 0;
        this.currentX = 0;
        this.currentY = 0;
        if (this.wInputRef) this.wInputRef.value = '';
        if (this.hInputRef) this.hInputRef.value = '';
        if (this.xInputRef) this.xInputRef.value = '';
        if (this.yInputRef) this.yInputRef.value = '';
        this.drawCropArea();
        this.updateNodeParameters();
        if (this.node) {
            api.fetchApi('/image_cropper/clear', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ node_id: this.nodeKey ?? this.node.id })
            });
        }
    }

    applySelection() {
        if (!this.node || !this.imageData) return;
        const rect = this.getImageRect();
        rect.x = Math.max(0, Math.min(rect.x, this.imageData.width - 1));
        rect.y = Math.max(0, Math.min(rect.y, this.imageData.height - 1));
        rect.width = Math.max(1, Math.min(rect.width, this.imageData.width - rect.x));
        rect.height = Math.max(1, Math.min(rect.height, this.imageData.height - rect.y));
        api.fetchApi('/image_cropper/apply', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: this.nodeKey ?? this.node.id,
                x: rect.x,
                y: rect.y,
                width: rect.width,
                height: rect.height,
            })
        }).then(() => {
            if (this.node && this.node.visualCropDialog) {
                try {
                    document.body.removeChild(this.node.visualCropDialog);
                } catch {}
                this.node.visualCropDialog = null;
            }
        });
    }

    getImageRect() {
        const x = Math.min(this.startX, this.currentX);
        const y = Math.min(this.startY, this.currentY);
        const width = Math.abs(this.currentX - this.startX);
        const height = Math.abs(this.currentY - this.startY);
        const imgX = Math.round((x - this.offsetX) / this.scale);
        const imgY = Math.round((y - this.offsetY) / this.scale);
        const imgWidth = Math.round(width / this.scale);
        const imgHeight = Math.round(height / this.scale);
        return { x: Math.max(0, imgX), y: Math.max(0, imgY), width: Math.max(1, imgWidth), height: Math.max(1, imgHeight) };
    }

    onImageInputConnected(imageUrl) {
        if (imageUrl) {
            this.loadImage(imageUrl);
        }
    }
}

// 注册ComfyUI扩展
app.registerExtension({
    name: "ImageAreaCrop.VisualCrop",
    async setup() {
        // 自动弹窗事件监听（由后端节点发送）
        api.addEventListener('image_area_crop_update', ({ detail }) => {
            const { node_id, image_data } = detail;
            const resolvedId = typeof node_id === 'string' ? parseInt(node_id, 10) : node_id;
            const node = app.graph.getNodeById(resolvedId);
            if (!node) return;
            if (!node.visualCropExtension) {
                const extension = new ImageAreaCropExtension();
                extension.bindNode(node);
                node.visualCropExtension = extension;
            }
            // 记录后端发来的唯一标识，确保提交与后端一致
            node.visualCropExtension.nodeKey = node_id;
            node.openVisualCropDialog?.();
            node.visualCropExtension.loadImage(image_data);
        });
    },
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "ImageAreaCropNode") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                const ret = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                // 创建可视化扩展实例
                const extension = new ImageAreaCropExtension();
                extension.bindNode(this);
                
                // 使用ComfyUI标准方式创建自定义widget
                const visualWidget = ComfyWidgets["STRING"](
                    this,
                    "visual_crop_interface",
                    ["STRING", {
                        multiline: true,
                        default: "点击右键选择 '打开可视化裁切界面' 来使用可视化选择功能",
                        placeholder: "可视化裁切界面"
                    }],
                    app
                );
                
                // 设置widget为只读
                if (visualWidget && visualWidget.widget) {
                    visualWidget.widget.inputEl.readOnly = true;
                    visualWidget.widget.inputEl.style.backgroundColor = "#2a2a2a";
                    visualWidget.widget.inputEl.style.color = "#ffffff";
                    visualWidget.widget.inputEl.style.border = "1px solid #555";
                }
                
                // 存储扩展实例
                this.visualCropExtension = extension;

                // 在节点上增加清除参数按钮
                this.addWidget("button", "清除参数", null, async () => {
                    try {
                        const clearId = this.visualCropExtension?.nodeKey ?? this.id;
                        await api.fetchApi("/image_cropper/clear", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ node_id: clearId })
                        });
                        if (this.visualCropExtension) {
                            this.visualCropExtension.hasSelection = false;
                            this.visualCropExtension.startX = 0;
                            this.visualCropExtension.startY = 0;
                            this.visualCropExtension.currentX = 0;
                            this.visualCropExtension.currentY = 0;
                        }
                    } catch (e) {
                        console.error("清除裁切参数失败", e);
                    }
                });

                // 添加右键菜单选项
                const originalGetExtraMenuOptions = this.getExtraMenuOptions;
                this.getExtraMenuOptions = function(_, options) {
                    if (originalGetExtraMenuOptions) {
                        originalGetExtraMenuOptions.apply(this, arguments);
                    }
                    
                    options.push({
                        content: "打开可视化裁切界面",
                        callback: () => {
                            this.openVisualCropDialog();
                        }
                    });
                };
                
                // 添加打开可视化界面的方法
                this.openVisualCropDialog = function() {
                    if (this.visualCropDialog) {
                        this.visualCropDialog.close();
                    }
                    
                    // 创建模态对话框
                    const dialog = document.createElement('div');
                    dialog.className = 'visual-crop-dialog';
                    dialog.style.cssText = `
                        position: fixed;
                        top: 50%;
                        left: 50%;
                        transform: translate(-50%, -50%);
                        background: #2a2a2a;
                        border: 2px solid #555;
                        border-radius: 8px;
                        padding: 20px;
                        z-index: 10000;
                        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
                    `;
                    
                    // 添加标题
                    const title = document.createElement('h3');
                    title.textContent = '可视化裁切界面';
                    title.style.cssText = `
                        color: #ffffff;
                        margin: 0 0 15px 0;
                        text-align: center;
                    `;
                    dialog.appendChild(title);
                    
                    // 创建可视化界面
                const visualInterface = this.visualCropExtension.createVisualInterface();
                if (visualInterface) {
                    dialog.appendChild(visualInterface);
                }

                // 获取已缓存图像（上次运行保存）以便右键打开能显示图像
                try {
                    api.fetchApi('/image_cropper/get', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ node_id: this.visualCropExtension.nodeKey ?? this.id })
                    }).then(async (res) => {
                        const data = await res.json();
                        if (data && data.image_data) {
                            this.visualCropExtension.loadImage(data.image_data);
                        }
                    }).catch(() => {});
                } catch {}
                    
                    // 添加关闭按钮
                    const closeBtn = document.createElement('button');
                    closeBtn.textContent = '关闭';
                    closeBtn.style.cssText = `
                        background: #666;
                        color: white;
                        border: none;
                        padding: 8px 16px;
                        border-radius: 4px;
                        cursor: pointer;
                        margin-top: 15px;
                        display: block;
                        margin-left: auto;
                        margin-right: auto;
                    `;
                    closeBtn.onclick = async () => {
                        try {
                            if (this.visualCropExtension && this.visualCropExtension.hasSelection) {
                                await api.fetchApi('/image_cropper/apply', {
                                    method: 'POST',
                                    headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify({
                                        node_id: this.visualCropExtension.nodeKey ?? this.id,
                                        ...this.visualCropExtension.getImageRect()
                                    })
                                });
                            } else {
                                await api.fetchApi('/image_cropper/cancel', {
                                    method: 'POST',
                                    headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify({ node_id: this.visualCropExtension.nodeKey ?? this.id })
                                });
                            }
                        } catch {}
                        document.body.removeChild(dialog);
                        this.visualCropDialog = null;
                    };
                    dialog.appendChild(closeBtn);
                    
                    // 添加到页面
                    document.body.appendChild(dialog);
                    this.visualCropDialog = dialog;
                };
                
                // 监听图像输入连接
                const originalOnConnectionsChange = this.onConnectionsChange;
                this.onConnectionsChange = function(type, slotIndex, isConnected, link, ioSlot) {
                    if (originalOnConnectionsChange) {
                        originalOnConnectionsChange.apply(this, arguments);
                    }
                    
                    if (type === 1 && slotIndex === 0 && isConnected) { // 输入连接
                        // 这里可以处理图像输入连接事件
                        if (this.visualCropExtension) {
                            // 获取连接的图像数据并加载
                            // this.visualCropExtension.onImageInputConnected(imageUrl);
                        }
                    }
                };
                
                return ret;
            };
        }
    }
});
