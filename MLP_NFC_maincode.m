%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MLP Neuro Fuzzy Control Project : V4.1 :
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%  Author : Ashkan Yousefi Zadeh /University of Guilan 
%%%%  Professor : Dr.Ali Jamali / University of Guilan 
%%%%  E_mail: a.yousefizadeh.edu@gmail.com
%%%%% Linkedin : www.linkedin.com/in/ashkan-ysf/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
clear ;
close all;
%%

fprintf('Enter Type of data that you want to working with = \n ');

fprintf(2,'(1) Manually import data /(2) xor example /(3)binary example/(4)diabetes csv/(5)advertisment csv = \n\n ');
ex=input('');

if ex==1 
    
    
    fprintf(2,'***hint***:> XOR example : X = [0 0 1 1;0 1 0 1];F = [0 1 1 0] \n\n');
    
    fprintf(2,'Enter your training data : \n\n');
    X=input('X=');
    
    fprintf(2,'Enter your training targets: \n');
    F=input('F=');
    
    fprintf(2,'Enter your testing data = \n');
    xtest=input('Xtest=');
    
    fprintf(2,'Enter your testing targets = \n');
    ytest=input('Ytest=');
    
elseif ex==2 
    
    run xor_ex.m;
    
elseif ex==3
    run data1_ex.m;

elseif  ex==4 
    run csvdata_ex.m;
    
elseif  ex==5 
    
    run csvdata2_ex.m;
    
elseif ex==6
    run csvdata3_ex.m;

end

%%

fprintf(2,'Enter number of hidden layers (1 or 2) =  ');
L=input('');

if L==1    
    fprintf(2,'Enter number of neurons in the hidden layer =  ');
    p1=input('');
    
else 
    fprintf(2,'Enter number of neurons in the first hidden layer =  ');
    p1=input(''); 
    
    fprintf(2,'Enter number of neurons in the second hidden layer =  ');
    p2=input('');
    
end



fprintf(2,'Enter the leaning rate =  ');
alpha=input('');

fprintf(2,'Enter the maxiumum Epoch to reach =  ');
epoch=input('');

fprintf(2,'Enter the minimum MSE to reach =  ');
MSEmin=input('');

fprintf(2,'Enter "1" for sigomid activation function,"2" for hyperbolic activation function,"3" for Relu activation function: \n');
Actype=input('Enter your activation function type  = ');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
n =size(X,1);
m =1;

if L==1 
    
    [W1,W2,MSE]=TrainMLP(n,p1,m,alpha,X,F,Actype,epoch,MSEmin);
    Y_train = TestMLP(X,W1,W2);
    Y_test=TestMLP(xtest,W1,W2);  
    

    
else 
    
    [W1,W2,W3,MSE]=TrainMLP2(n,p1,p2,m,alpha,X,F,Actype,epoch,MSEmin);
    Y_train = TestMLP2(X,W1,W2,W3);
    Y_test=TestMLP2(xtest,W1,W2,W3);
    
end


%%
    disp(['F = [' num2str(F) ']']);
    disp(['Y_train = [' num2str(Y_train) ']']);
    disp(['Ftest = [' num2str(ytest) ']']);
    disp(['Y_test= [' num2str(Y_test) ']']);
    train_Result=[Y_train',F'] % training Result
    test_Result=[Y_test',ytest']% testing Result
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%

figure('Name','Ashkan Yousefizadeh MLP(with 1 and 2 hidden layer)');


subplot(3,2,1);
histogram(X,10);
title('Train Inputs Histogram');
xlabel('Input Value');
ylabel('distribution');
%
subplot(3,2,5);
semilogy(MSE,'r','LineWidth',1);
title('Mean Square Error');
xlabel('Epochs');
ylabel('MSE');
grid on
%
subplot(3,2,3);
histogram(xtest,10);
title('Test Inputs Histogram');
xlabel('Input Value');
ylabel('distribution');

subplot(3,2,2);
t=linspace(1,size(F,2),size(F,2));
scatter(F,t,'filled','k','LineWidth',1.5);
hold on
scatter(Y_train,t,'LineWidth',1.5);

title('Train Validation');
ylabel('data Value');
legend('Y Train','Y Train predicted','Location','best')
grid on

subplot(3,2,4);
t=linspace(1,size(ytest,2),size(ytest,2));
scatter(ytest,t,'filled','k','LineWidth',1.5);
hold on
scatter(Y_test,t,'LineWidth',1.5);

title('Test Validation');
ylabel('data Value');
legend('Y Test','Y Test predicted','Location','best')
grid on
