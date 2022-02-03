
clear
clc
close all

%This scritp computes averaged ERP from two or more electrode sites to test
%for encoding and compares with non-encoding.
addpath(genpath('/work/TIBIR/s201302'))


% subj_sessions = {'CC016','UT004a','UT022','UT023','UT049','UT050','UT060','UT063',...
%     'UT064','UT067','UT069','UT074','UT081','UT090','UT095','UT105','UT108','UT112','UT113','UT117',...
%     'UT119','UT122','UT123','UT126','UT128','UT130','UT132','UT151'};

subj_sessions = {'UT008','UT009','UT011','UT012','UT013','UT014','UT015','UT016','UT017',...
    'UT018','UT019','UT020','UT021','UT022','UT025','UT026','UT027','UT028','UT029','UT031','UT032',...
    'UT033','UT034','UT035','UT039','UT040','UT041','UT043','UT045','UT049','UT050','UT052',...
    'UT053','UT055','UT056','UT058','UT061','UT062','UT063','UT064','UT065','UT066','UT067','UT068',...
    'UT087','UT088','UT090','UT092','UT093','UT095','UT096','UT097','UT108','UT116','UT121',...
    'UT123','UT126','UT128','UT130','UT0132'};
% subj_sessions =  {'UT108','UT116','UT121',...
%     'UT123','UT126','UT128','UT130','UT0132'}; 
%subj_sessions = {'UT069'};
RegionList  = {'AH-L'};
elecList = GetChannelList(RegionList,subj_sessions);
eleclen = cellfun(@length,elecList);
eleclen = min(eleclen');
acc=1;
% for subind = 1:length(subj_sessions)
%     Thissubj = elecList(subind,:);
%     Thissubj = cellfun(@(x)x(1:eleclen(subind)),Thissubj,'UniformOutput',false);
%     elecList(subind,:) = Thissubj;
% end

for rind =1:1
    tic
offset = 0;
acc1= 1;
acc2 =1;
for subind = 1:length(subj_sessions)
    Thissubj = subj_sessions{subind};
    
    
     
    events2load = ['/work/TIBIR/s201302/EEGDataset/FR1events/',Thissubj,'.mat'];
    load(events2load)   
    events = filterStruct(events,{'~strcmp(eegfile,'''')'}); %this does nothing
          
    for k = 1:length(events)
        str = events(k).eegfile;
        str_ind = strfind(str,Thissubj);
        if ~isempty(str_ind)
            fn = str(str_ind(2):end);
            if exist (['/project/TIBIR/Lega_lab/shared/lega_ansir/subjFiles/' Thissubj '/eeg.noreref/' fn,'.001'],'file')==2
            events(k).eegfile = ['/project/TIBIR/Lega_lab/shared/lega_ansir/subjFiles/' Thissubj '/eeg.noreref/' fn];
            else 
            events(k).eegfile = ['/project/TIBIR/Lega_lab/shared/lega_ansir/subjFiles/' Thissubj '/eeg.bipolar/' fn]; 
            end
        end
    end
        
     Fs = 1000; %sampling frequency
     N = 1800;    
     encEvents = filterStruct(events,{'recalled==1'});
     nencEvents = filterStruct(events,{'recalled==0'});     
     
    for Eleind = 1:length(elecList{subind,rind})
    
      ThisChan = elecList{subind,rind}(Eleind);    
      [encEEG1] = gete_ms(ThisChan, encEvents, N,offset);
      [nencEEG1] = gete_ms(ThisChan, nencEvents, N,offset);
      encEEG1 =  resample(encEEG1',1,4)';
      nencEEG1 = resample(nencEEG1',1,4)';
      % normalize and notch fitler EEG
      encEEG1 = encEEG1 - mean(encEEG1')';
      encEEG1 = notch_mult(encEEG1);
      encEEG1 = (encEEG1'./std(encEEG1'))';     
      nencEEG1 = nencEEG1 - mean(nencEEG1')';
      nencEEG1 = notch_mult(nencEEG1);
      nencEEG1 = (nencEEG1'./std(nencEEG1'))';
    

        bse_post = [];
        bsne_post = [];

        max_encpost = [];max_nencpost=[];max_encpostval=[];max_nencpostval=[];

        for k = 1:length(encEvents)
            [bse_post, w] = bispecd([encEEG1(k,:)'],256);
            QPC_recalled(acc1,:,:)= (abs(bse_post(128-64+1:128,128+1:128+64)));
            %QPC_recalled(acc1,:,:) = abs(bse_post(512:530,563:625));          
            labelRe(acc1) = 1;             
            acc1=acc1+1;
        end
%         imagesc((abs(bse_post(128-64+1:128,128+1:128+64))))
%         colorbar
%         %caxis([-2 4])
%         figure
%         imagesc((abs(bsne_post(128-64+1:128,128+1:128+64))))
%         colorbar
        %caxis([-2 4])
        %imagesc(zscore(abs(bsne_post)))
        
        for k = randi(length(nencEvents),1,length(encEvents))
            [bsne_post, w] = bispecd([nencEEG1(k,:)'],256);
            
            QPC_nonrecalled(acc2,:,:)= (abs(bsne_post(128-64+1:128,128+1:128+64)));
            %QPC_nonrecalled(acc2,:,:) = abs(bsne_post(512:530,563:625));
            labelNR(acc2) = 0;
            acc2=acc2+1;
            
        end

    end
    Thisfeature = [ QPC_recalled; QPC_nonrecalled;];
    labels = [labelRe labelNR];
    save([RegionList{rind},'-QPC.mat'],'Thisfeature','-v7.3')
    save([RegionList{rind},'-labels.mat'],'labels')
    
end
    toc
    clear QPC_recalled QPC_nonrecalled labelRe labelNR
end
