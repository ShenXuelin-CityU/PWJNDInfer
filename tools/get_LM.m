function [JND_LM] = get_LM(inputpatch)
Back_LM=mean(mean(inputpatch));
if Back_LM<127
    JND_LM=17*(1-sqrt(Back_LM/127))+3;
else
    JND_LM=3*(Back_LM-127)/128+3;
end
end

