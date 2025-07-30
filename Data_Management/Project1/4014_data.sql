COPY "4014_Characters" (ID, characterName, actorName0, actorName1, houseName, nickname, royal, Is_animal_Flag)
    FROM 'C:\Users\Γιώργος Μπόζιακας\Python_examples\DS_Test\Data_Management\Characters_master_data.csv'
    WITH (FORMAT csv, HEADER true);
COPY "4014_Character_Relationships" (source_character_ID, target_character_ID, relationship_type)
    FROM 'C:\Users\Γιώργος Μπόζιακας\Python_examples\DS_Test\Data_Management\Characters_relationships_data.csv'
    WITH (FORMAT csv, HEADER true);
COPY "4014_Episodes" (ID, seasonNum, episodeNum, episodeTitle, episodeAirDate, episodeDescription, eplength)
    FROM 'C:\Users\Γιώργος Μπόζιακας\Python_examples\DS_Test\Data_Management\Episodes_Master_data.csv'
    WITH (FORMAT csv, HEADER true);
COPY "4014_Episodes_Scenes" (ID, Scene, Start_Time, End_Time)
    FROM 'C:\Users\Γιώργος Μπόζιακας\Python_examples\DS_Test\Data_Management\Episodes_Scene_Times_data.csv'
    WITH (FORMAT csv, HEADER true);
COPY "4014_Locations" (ID, location)
    FROM 'C:\Users\Γιώργος Μπόζιακας\Python_examples\DS_Test\Data_Management\Location_Master_Data.csv'
    WITH (FORMAT csv, HEADER true);
COPY "4014_SubLocations" (ID, Sublocation_Name)
    FROM 'C:\Users\Γιώργος Μπόζιακας\Python_examples\DS_Test\Data_Management\Location_Sublocation.csv'
    WITH (FORMAT csv, HEADER true);
COPY "4014_Episodes_Scene_Characters" (ID, Scene, Char_id)
    FROM 'C:\Users\Γιώργος Μπόζιακας\Python_examples\DS_Test\Data_Management\Episodes_Characters_data.csv'
    WITH (FORMAT csv, HEADER true);
COPY "4014_Episodes_Scene_Locations_Sublocations" (ID, Scene, Loc_id, Sub_Name_y)
    FROM 'C:\Users\Γιώργος Μπόζιακας\Python_examples\DS_Test\Data_Management\Episodes_LocSubloc_data.csv'
    WITH (FORMAT csv, HEADER true);
COPY "4014_Opening_Locations" (ID, Sub_Type, Sub_Name)
    FROM 'C:\Users\Γιώργος Μπόζιακας\Python_examples\DS_Test\Data_Management\Episodes_Opening_Loc_data.csv'
    WITH (FORMAT csv, HEADER true);