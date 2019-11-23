clear
%numTiles = [3000, 4000, 2000];
% some parameters

% as defined in the paper the minimal distance should be 200 not 100
%minDist = 100;
minDist = 200;
d2 = minDist^2;

wait = 0;
wait_per_slide = 1000000000000;
generate_train_patches = 1;

tileSize = 512;
logging = {};
logNr = 1;
dirs = ["UBC", "UMch"];
%dirs = ["UMch"];
savepath = '/Volumes/VERBATIMHD/Code/data/semantic_segmentation_histo/epidermis_segmentation_kay/extracted_patches_dataset';
tileFolder = 'patches';
labelFolder = 'label';
if ~exist(fullfile(savepath, tileFolder), 'dir')
    
    mkdir(fullfile(savepath, tileFolder))
end

if ~exist(fullfile(savepath, labelFolder), 'dir')
    mkdir(fullfile(savepath, labelFolder))
end


%% Get all test slide names
formatSpec = '%s';
fileID = fopen('/Users/jeremyscheurer/Code/semantic-segmentation-of-histopathological-slides/tools/oskal_etal_dataset_tools/test_slides.txt','r');
test_slide_names = textscan(fileID,'%s','delimiter','\n'); 

%%

%loop over both datasets
for j = length(dirs):-1:1 
    % Select case location (UBC, SUS or UMch)
    fprintf('Extractig image tiles and labels from %s database \n', dirs(j))
    fprintf('**************************************************\n\n')
    curr_db = dirs(j);
    caseDir = sprintf('/Volumes/VERBATIMHD/Code/data/semantic_segmentation_histo/epidermis_segmentation_kay/%s', curr_db);
    cases = dir(fullfile(caseDir, '*.tif'));
    cases = {cases.name};
    
    if generate_train_patches
        for z = 1:length(test_slide_names)
            if any(ismember(cases, test_slide_names{1}{z}))
                cases(ismember(cases, test_slide_names{1}{z})) = [];
            end
        end
    else 
        cases = intersect(cases, test_slide_names{1});
    end
    
    % Iterate through cases
    for i = 1:length(cases)
        % Remove file-extention
        [~,caseName,~] = fileparts(cases{i});
        
        logging{logNr, 1} = caseName;
        
        fprintf('Case number: %s\n-----------------------\n', caseName);
        svs_file = sprintf('%s/%s.tif', caseDir, caseName);
        GT_file = sprintf('%s/ground_truth/%s_gt.tif', caseDir, caseName);
        svsinfo = imfinfo(svs_file);
        
        %********** Load and resize data ***************
        if exist(GT_file, 'file')
            tic
            GT = imread(GT_file);
            I = imread(svs_file, 'Index', 1);
            t = toc;
            fprintf('WSI loading time: %.2f seconds.\n\n', t)
            
            imgSize = size(I); % Size of original imagae
            GT = imresize(GT, [imgSize(1) imgSize(2)]);
            
            % Calculate number of tiles from image size (round to nearest 50)
            % afaik this calulates the number of ones for each dimension
            % i.e. for each class
            pixelsPerClass = sum(sum(GT));
            epidermisPixels = pixelsPerClass(3);
            numTiles = round((epidermisPixels/(512^2)*10)/50)*50;
            numTiles = numTiles * 3;
            fprintf('For this slide we are creating %d tiles', numTiles)
            logging{logNr, 2} = numTiles;
            
            %********** Generate random tiles ***************
            points = [0 0]; % Starting point for distance checking
            idx = 0;
            wait_idx = 0;
            nEpidermis = 0;
            nBackground = 0;
            nOtherTissue = 0;
            
            h1 = waitbar(0,'Initializing waitbar...','Name',sprintf('Extract tiles from %s', caseName));
            while (size(points, 1) <= numTiles)
                wait_idx = wait_idx +1;
                if wait_idx > wait_per_slide && wait
                    fprintf('breaking loop')
                    break
                end
                % Randomly generate a new upper left point
                point = floor(rand(1, 2) .* [imgSize(2) imgSize(1)]);
                
                % The corners for the tile
                y1 = point(2);
                y2 = point(2)+tileSize-1;
                x1 = point(1);
                x2 = point(1)+tileSize-1;
                
                % Check if x2 or y2 exceeds image size
                if x2 > imgSize(2) || y2 > imgSize(1) || x1 == 0|| y1 == 0
                    continue
                end
                
                % Calculate squared distances to all other points
                dist2 = sum((points - repmat(point, size(points, 1), 1)) .^ 2, 2);
                
                % Only add this point if it is far enough away from all others.
                if (all(dist2 > d2))
                    
                    % Count pixels per class
                    tileGT = GT(y1:y2, x1:x2, :);
                    classPixelRatio = sum(sum(tileGT))./sum(sum(sum(tileGT)));
                    
                    % Decide class
                    if classPixelRatio(:,:,3) > 0.4
                        % Check if max total number of epidermis tiles are reached
                        if nEpidermis < (numTiles/3)
                            nEpidermis = nEpidermis + 1;
                            class = 3;
                        else
                            continue
                        end
                        
                    elseif classPixelRatio(:,:,1) > 0.6
                        % Check if number of background tiles reached max
                        if nBackground < (numTiles/3)
                            nBackground = nBackground + 1;
                            class = 1;
                        else
                            continue
                        end
                        
                    elseif classPixelRatio(:,:,2) > 0.6
                        % Check if number of other tissue tiles reached max
                        if nOtherTissue < (numTiles/3)
                            nOtherTissue = nOtherTissue + 1;
                            class = 2;
                        else
                            continue
                        end
                    else
                        continue
                    end
                    
                    points = [points; point];
                else
                    continue
                end
                
                % Extract tile
                tileI = I(y1:y2, x1:x2, :);
                %classPixelRatio
                %imshow(imoverlay(tileI(:,:,1), tileGT(:,:,1), 'r'))
                %pause
                
                % Tile filename
                tileName = sprintf('%s_X%d_Y%d_class%d.png',caseName, x1, y1, class);
                tileName = fullfile(savepath, tileFolder, tileName);
                labeName = sprintf('%s_X%d_Y%d_class%d.png',caseName, x1, y1, class);
                labeName = fullfile(savepath, labelFolder, labeName);
                
                % Write image tile to disk
                imwrite(tileI, tileName);
                % Write image label (GT) to disk
                imwrite(uint8(255 * tileGT(:,:,3)), labeName);
                
                % Augment images
                % I think in our case we wont directly augment the images 
                %augI = imageAugmentation(tileI);
                %augGT = imageAugmentation(tileGT);
                %fields = fieldnames(augI);
                
                % Write augmented images to disk
                %for k = 1:length(fields)
                 %   augTileName = fullfile(savepath, tileFolder, ...
                  %      sprintf('%s_X%d_Y%d_%s.png', ...
                   %     caseName, x1, y1, fields{k}));
                    %augLabelName = fullfile(savepath, labelFolder, ...
                     %   sprintf('%s_X%d_Y%d_%s.png', ...
                      %  caseName, x1, y1, fields{k}));
                    %imwrite(augI.(fields{k}), augTileName);
                    %imwrite(augGT.(fields{k}), augLabelName);
                %end
                %clear augI augGT
                idx = idx + 1;
                wait_idx = 0;
                waitbar(idx/numTiles, h1, sprintf('Tile number %d written to disk', idx))
            end
            close(h1)
            logging{logNr, 3} = nBackground;
            logging{logNr, 4} = nOtherTissue;
            logging{logNr, 5} = nEpidermis;
            logNr = logNr + 1;
        else
            fprintf('Ground truth mask missing\n\n')
            logging{logNr, 2} = 0;
            logging{logNr, 3} = 0;
            logging{logNr, 4} = 0;
            logging{logNr, 5} = 0;            
            logNr = logNr + 1;
        end
        
    end
    
end
xlswrite('numTilesExtra.xls',logging)
