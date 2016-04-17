function [nc] = ncolumns(A)
%Returns the number of columns in A
temp = size(A);
nc = temp(2);
end
