% for every profile of speed
type = "sinusoid"; % change here
profiles = var_MagnetSinusoid.profile;
mkdir("I:\" + type)
fd_path = "I:\" + type;
cd(fd_path)

for i = 1 : length(profiles)
    mkdir("profile" + i)
    cd(fd_path + "\profile" + i)
    
    % for every fly inside each profile
    for j = 1 : length(profiles{i,1})
       data = profiles{i,1}(j).Data;
       writetable(data, "table" + j + ".csv")
    end
    cd(fd_path)
end
        