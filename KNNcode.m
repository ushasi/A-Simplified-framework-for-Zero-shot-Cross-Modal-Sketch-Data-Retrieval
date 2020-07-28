%new_feats = normc(fc_features)
featsx = photo_feat_un;
featsy = sketch_un;
y = randsample(14591 , 400 ); #random query samples
samples = y';
q =size(samples) 
t=64; %Dimension of the embedding space feature vectors

clear distance; clear unified_knn;clear unified_Ind
for i=1: q(2)       
    query = samples(i);
    

    for x=1:2499%14591  #unseen 25 class samples for Sketchy dataset
        sum=0;sum2=0;
        for y=1:t
            
            sum2 = sum2 + sqrt((featsx(x,y) -  featsy(query,y)) * (featsx(x,y) -  featsy(query,y))); % k-NN distance using l2 distance
            %sum2 = sum2 + (abs(new_feats(x,y) -  new_feats(query,y)) ); % k-NN distance using l1 distance
            %sum2 = sum2 + mahal(new_feats(x,:),new_feats(query,:))   % k-NN distance using Mahalanobis distance       
        end
        
        %sum2 = sqrt(sum2);
        distance(x,i) = sum2;           
    end
         
end
[unified_knn, unified_Ind] = sort(distance, 1); 

Ind = unified_Ind;

%------------------------------ Actual label and Predicted labels-------------------------------------
act = zeros(100,q(2));
pred = zeros(100,q(2));
for i=1:q(2)
    for j = 1:100%23833
        pred(j,i) = photo_lab(Ind(j,i)) - 99;
        act(j,i) = sketch_lab(samples(1,i)) - 99;
    end
    i;
end

%------------------------------------ Precision---------------------------------------------------------

%% precision
for n=1:10
    for i=1:q(2)
        tp = 0; fp = 0; preci = 0;
        for j= 1: n*10
            if (pred(j,i) == act(j,i) )
                tp = tp + 1;
            else fp = fp + 1;
            end
        end
    prec(i) = tp/(tp+fp);
    sum = mean(prec);
    end
    precision (n) = sum;%*100/n;

    n  
end

%-----------------------------------------Recall----------------------------------------------------
rec= zeros(1,q(2));
%% recall

for n=1:10
    tp = 0; total = 100; reca = 0;
    for i=1:q(2)
        tp = 0;  
        for j= 1: n*10
           if (pred(j,i) == act(j,i) )
                tp = tp + 1;
            end
        end
    rec(i) = tp/total;
    sum2 = mean(rec);
    end  
    recall(n) = sum2 ;%/190;
    n
end

%------------------------------------ mean Average Precision---------------------------------------------------------
%% MAP
map = mean(precision);

%% ANMRR
