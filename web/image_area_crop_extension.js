import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

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
        resetBtn.textContent = '重置';
        resetBtn.className = 'crop-btn';
        resetBtn.onclick = () => this.resetSelection();
        
        const applyBtn = document.createElement('button');
        applyBtn.textContent = '应用选择';
        applyBtn.className = 'crop-btn crop-btn-primary';
        applyBtn.onclick = () => this.applySelection();
        
        controls.appendChild(resetBtn);
        controls.appendChild(applyBtn);
        
        container.appendChild(this.canvas);
        container.appendChild(controls);
        
        // 绑定鼠标事件
        this.bindCanvasEvents();
        
        return container;
    }

    bindCanvasEvents() {
        if (!this.canvas) return;

        this.canvas.addEventListener('mousedown', (e) => {
            this.isDragging = true;
            const rect = this.canvas.getBoundingClientRect();
            this.startX = e.clientX - rect.left;
            this.startY = e.clientY - rect.top;
            this.currentX = this.startX;
            this.currentY = this.startY;
        });

        this.canvas.addEventListener('mousemove', (e) => {
            if (!this.isDragging) return;
            
            const rect = this.canvas.getBoundingClientRect();
            this.currentX = e.clientX - rect.left;
            this.currentY = e.clientY - rect.top;
            
            this.drawCropArea();
        });

        this.canvas.addEventListener('mouseup', () => {
            this.isDragging = false;
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

            // 显示尺寸信息
            this.ctx.fillStyle = '#00ff00';
            this.ctx.font = '12px Arial';
            this.ctx.fillText(`${Math.round(width)}x${Math.round(height)}`, x, y - 5);
        }
    }

    updateNodeParameters() {
        if (!this.node || !this.imageData) return;

        const x = Math.min(this.startX, this.currentX);
        const y = Math.min(this.startY, this.currentY);
        const width = Math.abs(this.currentX - this.startX);
        const height = Math.abs(this.currentY - this.startY);

        // 转换画布坐标到图像坐标
        const imgX = Math.round((x - this.offsetX) / this.scale);
        const imgY = Math.round((y - this.offsetY) / this.scale);
        const imgWidth = Math.round(width / this.scale);
        const imgHeight = Math.round(height / this.scale);

        // 更新节点参数
        const xWidget = this.node.widgets.find(w => w.name === 'x');
        const yWidget = this.node.widgets.find(w => w.name === 'y');
        const widthWidget = this.node.widgets.find(w => w.name === 'width');
        const heightWidget = this.node.widgets.find(w => w.name === 'height');

        if (xWidget) xWidget.value = Math.max(0, imgX);
        if (yWidget) yWidget.value = Math.max(0, imgY);
        if (widthWidget) widthWidget.value = Math.max(1, imgWidth);
        if (heightWidget) heightWidget.value = Math.max(1, imgHeight);
    }

    resetSelection() {
        this.startX = 0;
        this.startY = 0;
        this.currentX = 0;
        this.currentY = 0;
        this.drawCropArea();
        this.updateNodeParameters();
    }

    applySelection() {
        this.updateNodeParameters();
        // 可以在这里添加其他应用逻辑
        console.log('Selection applied');
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
                    closeBtn.onclick = () => {
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