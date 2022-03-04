%% Process dataset into mat files %%

clear;
clc;

%% Inputs:
% Locations of raw input files:
us101_1 = 'US-101/vehicle-trajectory-data/0750am-0805am/trajectories-0750am-0805am.txt';
us101_2 = 'US-101/vehicle-trajectory-data/0805am-0820am/trajectories-0805am-0820am.txt';
us101_3 = 'US-101/vehicle-trajectory-data/0820am-0835am/trajectories-0820am-0835am.txt';
i80_1 = 'I-80/vehicle-trajectory-data/0400pm-0415pm/trajectories-0400-0415.txt';
i80_2 = 'I-80/vehicle-trajectory-data/0500pm-0515pm/trajectories-0500-0515.txt';
i80_3 = 'I-80/vehicle-trajectory-data/0515pm-0530pm/trajectories-0515-0530.txt';


%% Fields: 

%{ 
1: Dataset Id
2: Vehicle Id
3: Frame Number
4: Local X
5: Local Y
6: Lane Id
7: Lateral maneuver
8: Longitudinal maneuver
9-47: Neighbor Car Ids at grid location
%}



%% Load data and add dataset id
disp('Loading data...')
traj{1} = load(us101_1);    
traj{1} = single([ones(size(traj{1},1),1),traj{1}]);
traj{2} = load(us101_2);
traj{2} = single([2*ones(size(traj{2},1),1),traj{2}]);
traj{3} = load(us101_3);
traj{3} = single([3*ones(size(traj{3},1),1),traj{3}]);
traj{4} = load(i80_1);    
traj{4} = single([4*ones(size(traj{4},1),1),traj{4}]);
traj{5} = load(i80_2);
traj{5} = single([5*ones(size(traj{5},1),1),traj{5}]);
traj{6} = load(i80_3);
traj{6} = single([6*ones(size(traj{6},1),1),traj{6}]);

for k = 1:6
    if k <=3
        traj{k}(traj{k}(:,15)>=6,15) = 6;
    end
end

%% Parse fields (listed above):
disp('Parsing fields...')

%% Split train, validation, test
disp('Splitting into train, validation and test sets...')

trajAll = [traj{1};traj{2};traj{3};traj{4};traj{5};traj{6}];
clear traj;

%% 

trajTr = [];
trajVal = [];
trajTs = [];
for k = 1:6
    ul1 = round(0.7*max(trajAll(trajAll(:,1)==k,2)));
    ul2 = round(0.8*max(trajAll(trajAll(:,1)==k,2)));
    
    trajTr = [trajTr;trajAll(trajAll(:,1)==k & trajAll(:,2)<=ul1, :)];
    trajVal = [trajVal;trajAll(trajAll(:,1)==k & trajAll(:,2)>ul1 & trajAll(:,2)<=ul2, :)];
    trajTs = [trajTs;trajAll(trajAll(:,1)==k & trajAll(:,2)>ul2, :)];
end

%% 

tracksTr = {};
for k = 1:6
    trajSet = trajTr(trajTr(:,1)==k,:);
    carIds = unique(trajSet(:,2));
    for l = 1:length(carIds)
        vehtrack = trajSet(trajSet(:,2) ==carIds(l),[3,6,7])';
        tracksTr{k,carIds(l)} = vehtrack;
    end
end

tracksVal = {};
for k = 1:6
    trajSet = trajVal(trajVal(:,1)==k,:);
    carIds = unique(trajSet(:,2));
    for l = 1:length(carIds)
        vehtrack = trajSet(trajSet(:,2) ==carIds(l),[3,6,7])';
        tracksVal{k,carIds(l)} = vehtrack;
    end
end

tracksTs = {};
for k = 1:6
    trajSet = trajTs(trajTs(:,1)==k,:);
    carIds = unique(trajSet(:,2));
    for l = 1:length(carIds)
        vehtrack = trajSet(trajSet(:,2) ==carIds(l),[3,6,7])';
        tracksTs{k,carIds(l)} = vehtrack;
    end
end


%% Filter edge cases: 
% Since the model uses 3 sec of trajectory history for prediction, the initial 3 seconds of each trajectory is not used for training/testing

disp('Filtering edge cases...')

indsTr = zeros(size(trajTr,1),1);
for k = 1: size(trajTr,1)
    t = trajTr(k,3);
    if tracksTr{trajTr(k,1),trajTr(k,2)}(1,31) <= t && tracksTr{trajTr(k,1),trajTr(k,2)}(1,end)>t+1
        indsTr(k) = 1;
    end
end
trajTr = trajTr(find(indsTr),:);


indsVal = zeros(size(trajVal,1),1);
for k = 1: size(trajVal,1)
    t = trajVal(k,3);
    if tracksVal{trajVal(k,1),trajVal(k,2)}(1,31) <= t && tracksVal{trajVal(k,1),trajVal(k,2)}(1,end)>t+1
        indsVal(k) = 1;
    end
end
trajVal = trajVal(find(indsVal),:);


indsTs = zeros(size(trajTs,1),1);
for k = 1: size(trajTs,1)
    t = trajTs(k,3);
    if tracksTs{trajTs(k,1),trajTs(k,2)}(1,31) <= t && tracksTs{trajTs(k,1),trajTs(k,2)}(1,end)>t+1
        indsTs(k) = 1;
    end
end
trajTs = trajTs(find(indsTs),:);

%% Save mat files:
disp('Saving mat files...')

traj = trajTr;
tracks = tracksTr;
save('TrainSet','traj','tracks');

traj = trajVal;
tracks = tracksVal;
save('ValSet','traj','tracks');

traj = trajTs;
tracks = tracksTs;
save('TestSet','traj','tracks');
