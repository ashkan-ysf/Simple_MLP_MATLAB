%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MLP Neuro Fuzzy Control Project : V4.1 :
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%  Author : Ashkan Yousefi Zadeh /University of Guilan 
%%%%  Professor : Dr.Ali Jamali / University of Guilan 
%%%%  E_mail: a.yousefizadeh.edu@gmail.com
%%%%% Linkedin : www.linkedin.com/in/ashkan-ysf/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function datascale = scaledata(datainput,minval,maxval)

datascale = datainput - min(datainput(:));
datascale = (datascale/range(datascale(:)))*(maxval-minval);
datascale = datascale + minval;
end