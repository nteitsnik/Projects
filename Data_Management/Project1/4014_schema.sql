Create table "4014_Characters"(
ID varchar(4) PRIMARY KEY ,
characterName varchar(50) NOT NULL,
actorName0 varchar(50),
actorName1 varchar(50),
houseName varchar(50),
nickname varchar(50),
royal boolean DEFAULT False,
Is_Animal_Flag boolean );

Create table "4014_Character_Relationships"(
source_character_ID varchar(4) REFERENCES "4014_Characters"(ID) NOT NULL, 
target_character_ID varchar(4) REFERENCES "4014_Characters"(ID) NOT NULL, 
relationship_type varchar(15)
 );

 Create table "4014_Locations"(
ID varchar(4) PRIMARY KEY, 
LOCATION VARCHAR(30) 
);

Create table "4014_SubLocations"(
ID varchar(4) REFERENCES  "4014_Locations"(ID) NOT NULL, 
Sublocation_Name VARCHAR(50) ,
PRIMARY KEY (ID,Sublocation_Name)
);
 
Create table "4014_Episodes"(
ID varchar(4) PRIMARY KEY, 
seasonNum varchar(2) NOT NULL ,
episodeNum  varchar(2) NOT NULL,
episodeTitle  varchar(50)NOT NULL,
episodeAirDate DATE NOT NULL,
episodeDescription varchar(240)  ,
eplength float );


Create table "4014_Episodes_Scenes"(
ID varchar(4) REFERENCES "4014_Episodes"(ID) NOT NULL, 
Scene integer  NOT NULL  ,
Start_Time  TIME , 
End_Time TIME  ,
PRIMARY KEY (ID,Scene)

);


Create table "4014_Episodes_Scene_Characters"(
ID varchar(4) REFERENCES "4014_Episodes"(ID) NOT NULL, 
Scene integer NOT NULL ,
Char_id  varchar(40)REFERENCES "4014_Characters"(ID)  NOT NULL, 
PRIMARY KEY (ID,Scene,Char_id) ,
FOREIGN KEY (ID , Scene) REFERENCES "4014_Episodes_Scenes"(ID, Scene)
);





Create table "4014_Episodes_Scene_Locations_Sublocations"(
ID varchar(4) REFERENCES "4014_Episodes"(ID) NOT NULL, 
Scene integer  NOT NULL ,
Loc_id  varchar(4) REFERENCES "4014_Locations"(ID) DEFAULT 'Unknown', 
Sub_Name_y varchar(50) DEFAULT 'Unknown',
PRIMARY KEY (ID,Scene),
FOREIGN KEY (Loc_id,Sub_Name_y) REFERENCES "4014_SubLocations"(ID,Sublocation_Name), 
FOREIGN KEY (ID , Scene) REFERENCES "4014_Episodes_Scenes"(ID, Scene)

);



Create table "4014_Opening_Locations"(
ID varchar(4) REFERENCES "4014_Episodes"(ID) NOT NULL, 
Sub_Type varchar(20) NOT NULL ,
Sub_Name   varchar(30),
PRIMARY KEY (ID,Sub_Type)

);
