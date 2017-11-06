function out = myclassify(data, filledInx)
%   Classifies

    %REMOVE empty arrays
    input=zeros(256,length(filledInx));
    input(:)=data(:,filledInx);


    %activation function
    %0-hardlim
    %1-linear
    %2-sigmoidal
    actFunc=load ('actFunc.mat');
    actFunc=actFunc.actFunc;
    
    %n select architecture
    %0-amc+classifier
    %1-single classifier
    n=load ('n.mat');
    n=n.n;
    
    %training(actFunc, n);

    switch n
        case 0
            %aml+cassifier
            A=AMC(input, actFunc);

        otherwise
            %classifier
            A=Single(input, actFunc);
    end

    [M,I] = max(A);
    I(find(M==0))=-1;
    out=I;
end

function A=AMC(Input,n)

    temp=load ('Wp.mat','-mat');
    Wp=temp.Wp;
    Input=Wp*Input;

    switch n
        case 0
            load AMChlNet;
            A=AMChlNet(Input);
        case 1
            load AMClinNet;
            A=AMClinNet(Input);
        otherwise
            load AMCsgNet;
            A=AMCsgNet(Input);
    end

end

function A=Single(Input, n)

    switch n
        case 0    
            load ShlNet;
            A=ShlNet(Input);
        case 1
            load SlinNet;
            A=SlinNet(Input);
        otherwise
            load SsgNet;
            A=SsgNet(Input);
    end

end

