%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MLP Neuro Fuzzy Control Project : V4.1 :
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%  Author : Ashkan Yousefi Zadeh /University of Guilan 
%%%%  Professor : Dr.Ali Jamali / University of Guilan 
%%%%  E_mail: a.yousefizadeh.edu@gmail.com
%%%%% Linkedin : www.linkedin.com/in/ashkan-ysf/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function AC = Act_func(z,select)
    if  nargin<2 
        AC = 1./(1+ exp(-z)); % sigmoid AC 
        
    elseif select == 1 
                AC = 1./(1+ exp(-z));
                
    elseif select == 2 
        
        AC=(exp(z)-exp(-z))/(exp(z)+exp(-z));
        
    elseif select == 3
        if z<0
        AC=0.001*z; %leaky relu 
        else 
            AC=z;
        end
     

    end
end


