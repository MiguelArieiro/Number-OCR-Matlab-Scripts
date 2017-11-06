function training (actFunc, n)

    temp=load('P.mat','-mat');
    P=temp.P;

    %activation function
    %0-hardlim
    %1-linear
    %2-sigmoidal

    %n select architecture
    %0-amc+classifier
    %1-single classifier

    switch n
        case 0
            %am+classifier
            TrainAMC(P, actFunc);
        otherwise
            %classifier
            TrainSingle(P, actFunc);
    end
end

function TrainAMC (P, n)
%AM+classifier

    temp=load('PerfectArial.mat','-mat');
    Perfect=temp.Perfect;

    %criar matriz T
    T=zeros(256,500);
    for num=1:10
        for r=1:50
            T(:,(num-1)*50+r)=Perfect(:,num);
        end
    end

    Wp=T*pinv(P); %weights
    save 'Wp.mat' Wp;

    switch n
        case 0    
            AMChlNet=hardlim(Perfect,eye(10));
            save AMChlNet;
        case 1
            AMClinNet=linear(Perfect,eye(10));
            save AMClinNet;
        otherwise
            AMCsgNet=sigmoidal(Perfect,eye(10));
            save AMCsgNet;
    end

end

function TrainSingle(P, n)
%Single NN Classifier

%criar target com base na mat identidade
    id=eye(10);
    T=zeros(10,500);
    for num=1:10
        for r=1:50
            T(:,(num-1)*50+r)=id(:,num);
        end
    end

    switch n
        case 0
            ShlNet=hardlim(P, T);
            save ShlNet;
        case 1
            SlinNet=linear(P, T);
            save SlinNet;
        otherwise
            SsgNet=sigmoidal(P, T);
            save SsgNet;
    end

end

%%Activation functions
function net = hardlim(P, T)
        
    hlNet=network;
    hlNet.numInputs=1;
    hlNet.inputs{1}.size=256;
    hlNet.numLayers=1;
    % assign the number of neurons in the layer
    hlNet.layers{1}.size=10;
    
    % connet input to layer 1
    hlNet.inputConnect(1)=1;
    % connect bias to layer 1
    hlNet.biasConnect(1)=1;
    hlNet.outputConnect(1)=1;
    
    % set layer transfer function
    hlNet.layers{1}.transferFcn='hardlim';
    
    %gradient method
    % set input weight learning function 
    hlNet.inputWeights{1}.learnFcn='learnp';
    % set bias learning function
    hlNet.biases{1}.learnFcn='learnp';

    % define the training function
    hlNet.trainFcn='trains';
    hlNet.performParam.lr = 0.5; % learning rate
    hlNet.trainParam.epochs = 1000; % maximum epochs
    hlNet.trainParam.show = 35; % show
    hlNet.trainParam.goal = 1e-6; % goal=objective
    hlNet.performFcn = 'mse'; % criterion 
    
    hlNet=train(hlNet,P,T);
    net=hlNet;

end

function net = linear(P, T)

    linNet=network;
    linNet.numInputs=1;
    linNet.inputs{1}.size=256;
    linNet.numLayers=1;
    % assign the number of neurons in the layer
    linNet.layers{1}.size=10;
    
    % connet input to layer 1
    linNet.inputConnect(1)=1;
    % connect bias to layer 1
    linNet.biasConnect(1)=1;
    linNet.outputConnect(1)=1;
    
    % set layer transfer function
    linNet.layers{1}.transferFcn='purelin';
    
    %gradient method
    % set input weight learning function 
    linNet.inputWeights{1}.learnFcn='learngd';
    % set bias learning function
    linNet.biases{1}.learnFcn='learngd';
    % define the training function
    linNet.trainFcn='trainscg';
    linNet.performParam.lr = 0.5; % learning rate
    linNet.trainParam.epochs = 1000; % maximum epochs
    linNet.trainParam.show = 35; % show
    linNet.trainParam.goal = 1e-6; % goal=objective
    linNet.performFcn = 'mse'; % criterion 
    
    linNet=train(linNet,P,T);
    net=linNet;
end


function net = sigmoidal(P, T)
    
    sgNet=network;
    sgNet.numInputs=1;
    sgNet.inputs{1}.size=256;
    sgNet.numLayers=1;
    % assign the number of neurons in the layer
    sgNet.layers{1}.size=10;
    
    % connet input to layer 1
    sgNet.inputConnect(1)=1;
    % connect bias to layer 1
    sgNet.biasConnect(1)=1;
    sgNet.outputConnect(1)=1;
    
    % set layer transfer function
    sgNet.layers{1}.transferFcn='logsig';
    
    %gradient method
    % set input weight learning function 
    sgNet.inputWeights{1}.learnFcn='learngd';
    % set bias learning function
    sgNet.biases{1}.learnFcn='learngd';

    % define the training function
    sgNet.trainFcn='trainscg';
    sgNet.performParam.lr = 0.5; % learning rate
    sgNet.trainParam.epochs = 1000; % maximum epochs
    sgNet.trainParam.show = 35; % show
    sgNet.trainParam.goal = 1e-6; % goal=objective
    sgNet.performFcn = 'mse'; % criterion 
    
    sgNet=train(sgNet,P,T);
    net=sgNet;
end

