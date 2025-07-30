# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 12:48:36 2024

@author: n.nteits
"""
import re
import pandas as pd
import os 
cwd = os.getcwd()
characters = pd.read_csv( r'Characters.csv',sep=';')
episodes = pd.read_csv( r'episodes.csv',sep=';')
dfk = pd.read_csv( r'keyValues.csv',sep=';')
locations = pd.read_csv( r'locations.csv',sep=';')

pd.set_option("display.max_columns", None)



characters['ID'] = range(len(characters))

characters = characters[['ID'] + [col for col in characters.columns if col != 'ID']]
episodes['ID'] = range(len(characters), len(characters) + len(episodes))
locations['ID'] = range(len(characters) + len(episodes), len(characters) + len(episodes) + len(locations))
# Move 'ID' column to the first position in episodes
episodes = episodes[['ID'] + [col for col in episodes.columns if col != 'ID']]

# Move 'ID' column to the first position in locations
locations = locations[['ID'] + [col for col in locations.columns if col != 'ID']]



# Move 'ID' column to the first position in characters
characters = characters[['ID'] + [col for col in characters.columns if col != 'ID']]

# Move 'ID' column to the first position in episodes
episodes = episodes[['ID'] + [col for col in episodes.columns if col != 'ID']]

# Move 'ID' column to the first position in locations
locations = locations[['ID'] + [col for col in locations.columns if col != 'ID']]

name_to_node_id = dict(zip(characters['characterName'], characters['ID'])) #Name-id-Driver Char
name_to_node_loc_id=dict(zip(locations['location'], locations['ID']))#Name-id-Driver Loc

# Create an empty DataFrame for Character_Relationships
character_relationships = pd.DataFrame(columns=['source_character_ID', 'target_character_ID', 'relationship_type'])

# Define relationship mappings
relationship_mappings = {
    'sibling': ['sibling0', 'sibling1', 'sibling2','sibling3','sibling4'],
    'parent': ['parent0', 'parent1', 'parent2'],
    'servedBy': ['servedBy0','servedBy1'],
    'married_engaged': ['marriedEngaged0', 'marriedEngaged1', 'marriedEngaged2'],
    'guardian_of': ['guardianOf0', 'guardianOf1', 'guardianOf2'],
    'servant_of': ['serves0','serves1', 'serves2', 'serves3'],
    'killed': ['killed0', 'killed1', 'killed2', 'killed3', 'killed4', 'killed5',
       'killed6', 'killed7', 'killed8', 'killed9', 'killed10', 'killed11',
       'killed12', 'killed13', 'killed14', 'killed15']
}

for relationship_type, columns in relationship_mappings.items():
    for column in columns:
        for index, row in characters.iterrows():
            source_id = row['ID']  # Source character's node_id
            target_name = row[column]  # Name of the related character
            if pd.notnull(target_name):  # Only process non-empty relationships
                target_id = name_to_node_id.get(target_name)
                if target_id:
                    # Append the relationship to the DataFrame
                    character_relationships = pd.concat(
                        [character_relationships, pd.DataFrame([{
                            'source_character_ID': source_id,
                            'target_character_ID': target_id,
                            'relationship_type': relationship_type.capitalize()  # e.g., Sibling, Parent
                        }])],
                        ignore_index=True
                    )


#Split Episodes CSV

#Character master data
#First 6 Columns
mcol=episodes.columns[0:6]                    
dfepisodeshdr=pd.DataFrame(data=episodes[mcol],columns=mcol)
print(dfepisodeshdr.head())

#Create Patterns in row names
patternnames = r"episodes/scenes/\d+/characters/\d+/name"
openingsec= r"episodes/openingSequenceLocations"
patternloc = r"episodes/scenes/\d+/location"
patternsubloc = r"episodes/scenes/\d+/subLocation"
patternstarttimes=r"episodes/scenes/\d+/sceneStart"
patternendtimes=r"episodes/scenes/\d+/sceneEnd"
#select Columns according to row Patterns
character_columns = [col for col in episodes.columns if re.match(patternnames, col)]
character_locations = [col for col in episodes.columns if re.match(patternloc, col)  ]
character_sublocations = [col for col in episodes.columns if re.match(patternsubloc, col) ]
character_times_start = [col for col in episodes.columns if re.match(patternstarttimes, col) ]
character_times_end= [col for col in episodes.columns if re.match(patternendtimes, col) ]
opening_sequence_locations=[col for col in episodes.columns if re.match(openingsec, col)]


#Create Dfs unpivoting columns of interest
#Chars on episodes
dfepisodeschar = pd.melt(
    episodes,
    id_vars=["ID"],  # 'Age' is omitted
    value_vars=character_columns,  # Columns to unpivot
    var_name="Type",  # Name for the new column with the unpivoted headers
    value_name="Char_Name"  # Name for the unpivoted values
).dropna(subset=["Char_Name"])
print(dfepisodeschar.head())

#Sublocations
unpivotedsubl = pd.melt(
    episodes,
    id_vars=["ID"],  
    value_vars=character_sublocations,  # Columns to unpivot
    var_name="Sub_Type",  # Name for the new column with the unpivoted headers
    value_name="Sub_Name"  # Name for the unpivoted values
)

#Locations
unpivotedloc = pd.melt(
    episodes,
    id_vars=["ID"], 
    value_vars=character_locations,  # Columns to unpivot
    var_name="Sub_Type",  # Name for the new column with the unpivoted headers
    value_name="Sub_Name"  # Name for the unpivoted values
)
#Opening Sequence Locations
dfunpivotedopeningloc = pd.melt(
    episodes,
    id_vars=["ID"], 
    value_vars=opening_sequence_locations,  # Columns to unpivot
    var_name="Sub_Type",  # Name for the new column with the unpivoted headers
    value_name="Sub_Name"  # Name for the unpivoted values
).dropna(subset=["Sub_Name"])

#Merge Loc and Subloc dfs
#Create key to merge
unpivotedsubl['Extra_key']=unpivotedsubl['Sub_Type'].str.extract(r"(.+)/[^/]+$")
unpivotedloc['Extra_key']=unpivotedloc['Sub_Type'].str.extract(r"(.+)/[^/]+$")
#Merge
dfeplocsubloc = pd.merge(unpivotedloc, unpivotedsubl, on=["ID", "Extra_key"]).drop(
    columns=[ "Sub_Type_x","Sub_Type_y"]
)
#Rearange columns
dfeplocsubloc=dfeplocsubloc[['ID','Extra_key','Sub_Name_x','Sub_Name_y']]
print(dfeplocsubloc.head())





#Starttimes
unpivotedst = pd.melt(
    episodes,
    id_vars=["ID"],  
    value_vars=character_times_start,  # Columns to unpivot
    var_name="Sub_Type",  # Name for the new column with the unpivoted headers
    value_name="Sub_Name"  # Name for the unpivoted values
)

#Endtimes
unpivotedet = pd.melt(
    episodes,
    id_vars=["ID"], 
    value_vars=character_times_end,  # Columns to unpivot
    var_name="Sub_Type",  # Name for the new column with the unpivoted headers
    value_name="Sub_Name"  # Name for the unpivoted values
)
unpivotedst['Extra_key']=unpivotedsubl['Sub_Type'].str.extract(r"(.+)/[^/]+$")
unpivotedet['Extra_key']=unpivotedloc['Sub_Type'].str.extract(r"(.+)/[^/]+$")

dfeptimes = pd.merge(unpivotedst, unpivotedet, on=["ID", "Extra_key"]).drop(
    columns=[ "Sub_Type_x","Sub_Type_y"]
)
dfeptimes=dfeptimes[['ID','Extra_key','Sub_Name_x','Sub_Name_y']]
print(dfeplocsubloc.head())

#Clean Columns of big text


#episodeshdr is ok
dfeptimes['Scene']=dfeptimes['Extra_key'].str.extract(r'(\d+)$')
dfeplocsubloc['Scene']=dfeplocsubloc['Extra_key'].str.extract(r'(\d+)$')
dfunpivotedopeningloc['Sub_Type_1']=dfunpivotedopeningloc['Sub_Type'].str.extract(r'(\d+)$')
dfepisodeschar['Scene']=dfepisodeschar['Type'].str.extract(r'(scenes/\d+)')


#Reorder
dfeptimes=dfeptimes.sort_values(by=["ID", "Extra_key"])
dfeplocsubloc=dfeplocsubloc.sort_values(by=["ID", "Extra_key"])#Need to change name with id--Locations
dfunpivotedopeningloc=dfunpivotedopeningloc.sort_values(by=["ID", "Sub_Type"])#Need to change name with id--Locations
dfepisodeschar=dfepisodeschar.sort_values(by=["ID", "Type"]) #Need to change name with id--Chars

#Names to ids
dfeplocsubloc['Loc_id']=dfeplocsubloc['Sub_Name_x'].replace(name_to_node_loc_id) #Add id subloc as new column 
#dfunpivotedopeningloc['Sub_id']=dfunpivotedopeningloc['Sub_Name'].replace(name_to_node_loc_id) #Add id subloc as new column 
dfepisodeschar['Char_id']=dfepisodeschar['Char_Name'].replace(name_to_node_id)
#Drop names columns
dfeplocsubloc.drop(columns=['Sub_Name_x'],inplace=True)
dfepisodeschar.drop(columns=['Char_Name'],inplace=True)

#Rearrange
dfeplocsubloc=dfeplocsubloc[['ID','Scene','Loc_id','Sub_Name_y']]


#Now the locations
#Names
dflocationnames=locations[['ID','location']]
#columns of sublocations
columnswithsublocations=locations.columns[2:]
#melt and sort
dfsubl = pd.melt(
    locations,
    id_vars=["ID"],  
    value_vars=columnswithsublocations,  # Columns to unpivot
    var_name="Sub_Type",  # Name for the new column with the unpivoted headers
    value_name="Sublocation_Name"  # Name for the unpivoted values
).dropna(subset=["Sublocation_Name"]).sort_values(by=['ID','Sub_Type'])

# Character name-ID
dfcharacters=characters[characters.columns[0:7]]
#Drop column Subtype
dfsubl.drop(columns=['Sub_Type'],inplace=True)
# Character name-ID
dfcharacters=characters[characters.columns[0:7]]

#Clean special characters 
dfcharacters['actorName0'] = dfcharacters['actorName0'].str.replace('Þ', 'P', regex=False)
dfcharacters['actorName0'] = dfcharacters['actorName0'].str.replace('ó', 'o', regex=False)
dfcharacters['characterName'] = dfcharacters['characterName'].str.replace('#', ' ', regex=False)
 #royal nans to false 
dfcharacters['royal']=dfcharacters['royal'].fillna(False)
#animals
animallist=['Drogon','Reageal','Viserion','Ghost','Grey Wind','Lady','Nymeria','Shaggydog','Summer']
dfcharacters.loc[:,'Is_animal_Flag']=dfcharacters['characterName'].isin(animallist)
#fix scenes in char scenes 
dfepisodeschar['Scene'] = dfepisodeschar['Scene'].str.extract(r'scenes/(\d+)')

#handle episode times  and drop dfk 
patterntimes=r"episodes/\d+/episodeTitle"
columns_to_drop = ['length'] +[col for col in dfk.columns if re.match(patterntimes, col)]
dfk1=dfk.drop(columns=columns_to_drop)
#Rename columns in dfepisode
dfepisodeshdr.columns = dfepisodeshdr.columns.str.replace('episodes/', '', regex=False)
dfepisode_times=pd.DataFrame(columns=['seasonNum','episodeNum','eplength'])
for i in range(1,len(dfk1.columns)-1,2):
    dftemp=dfk1[dfk1.columns[[0,i,i+1]]]
    dftemp.columns=dfepisode_times.columns
    dfepisode_times=pd.concat([dfepisode_times,dftemp],ignore_index=True)
   
dfepisodeshdr = pd.merge(dfepisodeshdr, dfepisode_times, on=['seasonNum', 'episodeNum'])

#Rename Columns for query creation
dfepisodeschar.drop('Type', axis=1, inplace=True)

dfepisodeshdr.columns = dfepisodeshdr.columns.str.replace('episodes/', '', regex=False)

#Rearrange columns
dfunpivotedopeningloc=dfunpivotedopeningloc[['ID','Sub_Type_1','Sub_Name']]
dfunpivotedopeningloc = dfunpivotedopeningloc.rename(columns={'Sub_Type_1': 'Sub_Type'})

dfeptimes=dfeptimes[['ID','Scene','Sub_Name_x','Sub_Name_y']]
dfeptimes = dfeptimes.rename(columns={'Sub_Name_x': 'Start_Time','Sub_Name_y': 'End_Time'})

dfepisodeschar['Char_id'] = pd.to_numeric(dfepisodeschar['Char_id'], errors='coerce')  # Convert non-numeric to NaN
dfepisodeschar = dfepisodeschar.dropna(subset=['Char_id'])  # Drop rows with NaN
dfepisodeschar['Char_id'] = dfepisodeschar['Char_id'].astype(int)


#export top csv

dfcharacters.to_csv(r".\Characters_master_data.csv", index=False) 
character_relationships.to_csv(r".\Characters_relationships_data.csv", index=False) 
dfepisodeshdr.to_csv(r".\Episodes_Master_data.csv", index=False) 
dfepisodeschar.to_csv(r".\Episodes_Characters_data.csv", index=False) 
dfeplocsubloc.to_csv(r".\Episodes_LocSubloc_data.csv", index=False) 
dfeptimes.to_csv(r".\Episodes_Scene_Times_data.csv", index=False)
dfunpivotedopeningloc.to_csv(r".\Episodes_Opening_Loc_data.csv", index=False)#needs care
dflocationnames.to_csv(r".\Location_Master_Data.csv", index=False)
dfsubl.to_csv(r".\Location_Sublocation.csv", index=False)




files_and_dfs = [
    ("Characters_master_data.csv", dfcharacters, '"4014_Characters"'),
    ("Characters_relationships_data.csv", character_relationships, '"4014_Character_Relationships"'),
    ("Episodes_Master_data.csv", dfepisodeshdr, '"4014_Episodes"'),
    ("Episodes_Scene_Times_data.csv", dfeptimes, '"4014_Episodes_Scenes"'),
    ("Location_Master_Data.csv", dflocationnames, '"4014_Locations"'),
    ("Location_Sublocation.csv", dfsubl, '"4014_SubLocations"'),
    ("Episodes_Characters_data.csv", dfepisodeschar, '"4014_Episodes_Scene_Characters"'),
    ("Episodes_LocSubloc_data.csv", dfeplocsubloc, '"4014_Episodes_Scene_Locations_Sublocations"'),
    ("Episodes_Opening_Loc_data.csv", dfunpivotedopeningloc, '"4014_Opening_Locations"'),
]

queries = []

# Iterate through files_and_dfs to generate SQL commands
for file_name, df, table_name in files_and_dfs:
    # Ensure column names are strings
    column_names = ', '.join(map(str, df.columns))
    
    # Generate SQL command
    sql = f"""
    COPY {table_name} ({column_names})
    FROM '{cwd}\\{file_name}'
    WITH (FORMAT csv, HEADER true);
    """
    
    queries.append(sql.strip())  # Strip extra whitespaces


with open("4014_data.sql", "w", encoding="utf-8") as f:
    f.write("\n".join(queries))