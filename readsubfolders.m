files = dir('ImageResized/');

label = zeros(204071,1);
 %{
% Get a logical vector that tells which is a directory.
dirFlags = [files.isdir];
% Extract only those that are directories.
subFolders = files(dirFlags);
subFolders(1:2) = [];
% Print folder names to command window.
for k = 1 : length(subFolders)
       fprintf('Sub folder #%d = %s\n', k, subFolders(k).name);
       %p=(classFolders(k).name)
       a = dir(['ImageResized/' subFolders(k).name '/*.*']);
       n(k) = numel(a)-2;
      
end
%}

ctr = 0;
for k = 1:250
    for i = 1:n(k)
        ctr = ctr + 1;
        label(ctr) = k;
    end
end

