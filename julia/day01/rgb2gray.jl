using Images

img = load("gray.jpg"); # 自动norm 0.～1.
height,width = size(img);
# img = channelview(img); # CHW 格式
out = zeros(UInt8,(height,width)); # 输出图像 0~255

# floor 向下取整 ，ceil 向上取整,round 四舍五入
type_convert(x)= begin
					x = floor(x*255.0);
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

function rgb2gray(img,out,gamma=0.5)
	for i = 1:height,j = 1:width
		out[i,j] = type_convert(img[i,j]^gamma);
	end
	return out
end

out = rgb2gray(img,out,0.5);

# 0~255 -> 0.~1.
# img_rgb = colorview(RGB, normedview(out));
img_gray = colorview(Gray, normedview(out));
# save
save("gray_gamma.jpg",img_gray);
