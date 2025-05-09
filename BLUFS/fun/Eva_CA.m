
function [accuracy, ConMtx] = Eva_CA(dataCluster,dataLabel)

nData = length( dataLabel );
nC = max(dataLabel);
E = zeros( nC, nC );
for m = 1 : nData
    i1 = dataCluster( m );
    i2 = dataLabel( m );
    E( i1, i2 ) = E( i1, i2 ) + 1;
end
ConMtx=E';
E=-E;
[C,~]=hungarian(E);
nMatch=0;
for i=1:nC
    nMatch=nMatch-E(C(i),i);
end

accuracy = nMatch/nData;
