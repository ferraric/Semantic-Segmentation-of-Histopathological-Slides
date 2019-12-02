function [] = createTrueMask(path_to_xml_file, path_to_image_file, path_to_epidermis_file, createEpidermisMask, createForegroundMask, showFigure)

if nargin == 0
    showFigure = 0;
    createEpidermisMask = 0;
    createForegroundMask = 1;
end

%% Parse XML file
xDoc = xmlread(path_to_xml_file);
Regions=xDoc.getElementsByTagName('Region'); % get a list of all the region tags
for regioni = 0:Regions.getLength-1
    Region=Regions.item(regioni);  % for each region tag
    
    %get a list of all the vertexes (which are in order)
    verticies=Region.getElementsByTagName('Vertex');
    xy{regioni+1}=zeros(verticies.getLength-1,2); %allocate space for them
    negROA(regioni+1) = str2num(Region.getAttribute('NegativeROA'));
    for vertexi = 0:verticies.getLength-1 %iterate through all verticies
        %get the x value of that vertex
        x=str2double(verticies.item(vertexi).getAttribute('X'));
        
        %get the y value of that vertex
        y=str2double(verticies.item(vertexi).getAttribute('Y'));
        xy{regioni+1}(vertexi+1,:)=[x,y]; % finally save them into the array
    end
    
end


% figure,hold all
% set(gca,'YDir','reverse'); %invert y axis
% for zz=1:length(xy)
%     plot(xy{zz}(:,1),xy{zz}(:,2),'LineWidth',4)
% end


%% Create epidermis mask from poly
if createEpidermisMask
    s=1; %base level of maximum resolution
    s2=3; % down sampling
    hratio=imgData.svsinfo(s2).Height/imgData.svsinfo(s).Height;  %determine ratio
    wratio=imgData.svsinfo(s2).Width/imgData.svsinfo(s).Width;
    
    nrow=imgData.svsinfo(s2).Height;
    ncol=imgData.svsinfo(s2).Width;
    mask=zeros(nrow,ncol); %pre-allocate a mask
    for zz=1:length(xy) %for each region
        smaller_x=xy{zz}(:,1)*wratio; %down sample the region using the ratio
        smaller_y=xy{zz}(:,2)*hratio;
        
        %make a mask and add it to the current mask
        %this addition makes it obvious when more than 1 layer overlap each
        %other, can be changed to simply an OR depending on application.
        if negROA(zz) == 1
            maskTemp = poly2mask( smaller_x,smaller_y,nrow,ncol);
            mask(maskTemp) = 0;
        else
            maskTemp = poly2mask( smaller_x,smaller_y,nrow,ncol);
            mask = mask + maskTemp; %mask | maskTemp;
        end
        %     figure,hold all
        %     imshow(mask)
        
    end
    caseMaskName = sprintf('%sEpidermis/%s_epidermis.png', imgData.caseDir,...
        imgData.caseName);
    imwrite(logical(mask), caseMaskName);
    epidermis_mask = mask;
    clear mask
end
%% Create foregroundmask
if createForegroundMask
    % Convert to HSV
    
     % as all the ground truth annotations are downlsampled we also need to
    % downsample the image. Also hsv conversion cant handle such a large
    % image. 
    
    slide_image = imread(path_to_image_file);
    epidermis_annotation = imread(path_to_epidermis_file);
    shape = size(epidermis_annotation);
    resized_slide_image = imresize(slide_image, [shape(1), shape(2)]);
    
    I_HSV = rgb2hsv(resized_slide_image);
    %GT = imgData.GT;
    % Split HSV channels
    iH = I_HSV(:,:,1);
    iS = I_HSV(:,:,2);
    iV = I_HSV(:,:,3);
    clear I_HSV
    
    % Otsu thresholding on H and S channel
    thH = imbinarize(iH, graythresh(iH));
    %figure; imshow(thH)
    thS = imbinarize(iS, graythresh(iS));
    %figure; imshow(thS)
    clear iH iS iV
    
    % Combine H and S mask.
    FG = thS & thH;
    
    % Remove small pixle areas in background and perform closing and filling holes.
    se = strel('disk', 20);
    FG = imclose(FG, se);
    FG = imfill(FG, 'holes');
    FG = imopen(FG, se);
    
    [path_to_image_folder, casename, ~] = fileparts(path_to_image_file);
    fgMaskName = sprintf('%s/Foreground_generated/%s_FG.png', path_to_image_folder,...
        casename);
    if ~exist(fullfile(sprintf('%s/Foreground_generated', path_to_image_folder)), 'dir')
        mkdir(fullfile(sprintf('%s/Foreground_generated', path_to_image_folder)))
    end
    imwrite(FG, fgMaskName);
    % figure; imshow(FG)
end
%%
if showFigure
    figure;
    colormask_FG = repmat(FG, [1 1 3]);
    colormask_epidermis = repmat(epidermis_mask, [1 1 3]);
    colormask_dermis = repmat(xor(FG, epidermis_mask), [1 1 3]);
    foreground_I = imgData.I;
    epidermis_I = imgData.I;
    dermis_I = imgData.I;
    foreground_I(~colormask_FG) = 255;
    epidermis_I(~colormask_epidermis) = 255;
    dermis_I(~colormask_dermis) = 255;
    figure;
    subplot(2, 2, 1)
    imshow(imgData.I)
    subplot(2, 2, 2)
    imshow(foreground_I)
    subplot(2, 2, 3)
    imshow(epidermis_I)
    subplot(2, 2, 4)
    imshow(dermis_I)
    %imshowpair(foreground_I, epidermis_I, 'montage')
end

