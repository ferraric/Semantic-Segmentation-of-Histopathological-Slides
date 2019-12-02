clear 
image_folder_path = '/Volumes/VERBATIMHD/Code/data/semantic_segmentation_histo/epidermis_segmentation_kay/UBC'; 
path_to_xml_files = '/Volumes/VERBATIMHD/Code/data/semantic_segmentation_histo/epidermis_segmentation_kay/UBC/Annotations/';
epidermmis_folder_path = '/Volumes/VERBATIMHD/Code/data/semantic_segmentation_histo/epidermis_segmentation_kay/UBC/Epidermis';
all_xml_files = dir(fullfile(path_to_xml_files, '*.xml'));
all_xml_files = {all_xml_files.name};

for i = 1:length(all_xml_files)
    xml_file = all_xml_files{i};
    path_to_xml_file = sprintf('%s%s', path_to_xml_files, xml_file);
    file_name = strsplit(xml_file, '.');
    image_path = sprintf('%s/%s.tif', image_folder_path, file_name{1});
    if exist(fullfile(image_path), 'file')
        epidermis_path = sprintf('%s/%s_epidermis.png', epidermmis_folder_path, file_name{1});
        createTrueMask(path_to_xml_file, image_path,epidermis_path, 0,1,0);
    end
end