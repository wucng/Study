using Images
using CUDAnative
using CuArrays
using CUDAdrv

dev = CuDevice(0); # 设置GPU
ctx = CuContext(dev);

img = load("cinque_terre_small.jpg"); # img加载时会自动做norm  0.~1.
height,width=size(img);
img = channelview(img); # CHW 格式
out = zeros(UInt8,(3,height,width)); # 0~255 ,CHW 格式

kenerl = [0. -1. 0.;
         -1. 4. -1.;
         0. -1. 0.];

# CPU -- > GPU
img_d = CuArrays.CuArray(img);
out_d = CuArrays.CuArray(out);
kenerl_d = CuArrays.CuArray(kenerl);

# floor 向下取整 ，ceil 向上取整,round 四舍五入
type_convert(x)= begin
					x = floor(x);
					# 截断到0.0～255.0范围
					if x<0.0
						x = 0.0;
					elseif x>255.0
						x = 255.0;
					else
						x = x;
					end
					return convert(UInt8,x);
				end

function conv2d(img,out,kenerl,k_size=(3,3))
    bx = blockIdx().x;
    by = blockIdx().y;
    tx = threadIdx().x;
    height = gridDim().y;
    width = gridDim().x;

    # piexl = 0.0;
    tmp = 0.0;
    for i = 1:k_size[1],j = 1:k_size[2]
        cur_i = by-floor(Int,k_size[1]/2)+i-1;
        cur_j = bx-floor(Int,k_size[2]/2)+j-1;
        if cur_i<1 || cur_i>height || cur_j<1 || cur_j>width
            piexl = 0.0;
        else
            piexl = 255.0*img[tx,cur_i,cur_j];
        end
        tmp += piexl*kenerl[i,j];
    end

    out[tx,by,bx] = type_convert(tmp);

    return nothing;
end

# 0.000084 s
@time @cuda threads=(3,1,1) blocks=(width,height,1)  conv2d(img_d, out_d, kenerl_d, size(kenerl));
out = Array(out_d); # GPU-->CPU

img_rgb = colorview(RGB, normedview(out)); # 必须从0～255 转成 0.～1.
save("RGB.jpg",img_rgb);

destroy!(ctx);