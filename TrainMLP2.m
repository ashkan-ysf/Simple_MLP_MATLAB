%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MLP Neuro Fuzzy Control Project : V4.1 :
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%  Author : Ashkan Yousefi Zadeh /University of Guilan 
%%%%  Professor : Dr.Ali Jamali / University of Guilan 
%%%%  E_mail: a.yousefizadeh.edu@gmail.com
%%%%% Linkedin : www.linkedin.com/in/ashkan-ysf/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
function [W1,W2,W3,MSE]=TrainMLP2(n,p1,p2,m,alpha,X,F,Actype,epochMax,target_MSE)

%% Default Parameters :
bias=-1;
MSE_mat = zeros(1,epochMax);
%%
col=size(X,2);

X = [bias*ones(1,col) ; X];


%% 
a=-0.3;
b=0.3;
W1=a+(b-a)*rand(p1,n+1);
W2=a+(b-a)*rand(p2,p1+1); % generating random numbers for Wx(inputs weights to Zp) -0.5<Wx<05
W3=a+(b-a)*rand(m,p2+1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Feed Forward :
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:epochMax
    j = 1:col;
    X = X(:,j);
    F = F(:,j);
    %%
     Z_in=W1*X;
    
     Z=Act_func(Z_in,Actype);
    
     Z2y = [bias*ones(1,col);Z]; %out put z to being input for y
        
     Y_in = W2*Z2y;
    
     Y=Act_func(Y_in,Actype);  
     
     Y2s=[bias*ones(1,col);Y];
     
     S_in=W3*Y2s;
     
     S=Act_func(S_in,Actype);  
     
     E = F - S ;% error for between computed Y and the real Y
    %%
switch Actype
    case 1        
%%
% Back Propagation case 1 :

% for the layer (S):
        fprim = S.*(1-S);
        
        delta_s = fprim .* E;
        
        Delta_Ws=alpha*delta_s*Y2s';
        
        W3 = W3 + Delta_Ws;
        
% for the layer (Y):
        fprim = Y2s.*(1-Y2s);
        
        delta_y = fprim .* (W3' * delta_s);
        
        delta_y = delta_y(2:end,:);
        
        Delta_Wy= alpha*delta_y*Z2y';
        
        W2 = W2 + Delta_Wy;

        
% for the 1st hidden layer(Z):
        fprim = Z2y.*(1-Z2y);

        delta_z = fprim .* (W2' * delta_y);
        
        delta_z = delta_z(2:end,:);
        
        Delta_Wz= alpha*delta_z*X';

        W1 = W1 + Delta_Wz;

    case 2
%% Back Propagation case 2:


% for the final layer (Y):

        fprim = 1-(Y^2);

        delta_y = fprim .* E;

        Delta_Wy=alpha*delta_y*Z2y';

        W2 = W2 + Delta_Wy;
       
    
% for the 1st hidden layer(Z):
        fprim =1-(Z2y^2);

        delta_z = fprim .* (W2' * delta_y);
        delta_z = delta_z(2:end,:);
        Delta_Wz= alpha*delta_z*X';

        W1 = W1 + Delta_Wz;
    
       
        
%%
    case 3
%% Back Propagation case 3:


% for the final layer (Y):
        if Y_in>0

        fprim = 1;

        delta_y = fprim .* E;

        Delta_Wy=alpha*delta_y*Z2y';

        W2 = W2 + Delta_Wy;
        else
        fprim = 0;

        delta_y = fprim .* E;
    
        Delta_Wy=alpha*delta_y*Z2y';

        W2 = W2 + Delta_Wy;
        
        end
    
% for the 1st hidden layer(Z):
       if Z_in>0
       fprim =1;

       delta_z = fprim .* (W2' * delta_y);
       delta_z = delta_z(2:end,:);
       Delta_Wz= alpha*delta_z*X';

       W1 = W1 + Delta_Wz;
    
       else
        
       fprim =0;
       delta_z = fprim .* (W2' * delta_y);
       delta_z = delta_z(2:end,:);
       Delta_Wz= alpha*delta_z*X';

       W1 = W1 + Delta_Wz;
        
    
      end


end
    %%
    mse =immse(F,S); %mean(mean(E.^2));
    
    MSE_mat(i) = mse;
    
    disp(['epoch = ' num2str(i) ' MSE = ' num2str(mse)]);
    
    if (mse < target_MSE)
        MSE = MSE_mat(1:i);
    return
    end
    

end

    MSE = MSE_mat;

end

