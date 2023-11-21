paths = ["A1", "A2", "J1", "J2", "psi", "particles"];

if ~isfolder(resultsPath)
   mkdir(resultsPath)
end
if ~isfolder(csvPath)
   mkdir(csvPath)
end
if ~isfolder(figPath)
   mkdir(figPath)
end
for idx = 1:length(paths)
    dir_path = csvPath + paths(idx);
    if ~isfolder(dir_path)
       mkdir(dir_path)
    end
end