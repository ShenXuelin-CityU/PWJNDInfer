function [sign] = get_randi()
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
di=randi(10);
    if di<=5
        sign=-1;
    else
        sign=1;
    end
end

