--i
CREATE INDEX character_id ON "4014_Characters" USING BTREE(id)
CREATE INDEX char_id ON "4014_Episodes_Scene_Characters" USING BTREE(char_id)

--ii
CREATE INDEX ep_id ON "4014_Episodes" USING BTREE(id)
CREATE INDEX ep_loc_id ON "4014_Episodes_Scene_Locations_Sublocations" USING BTREE(id)
CREATE INDEX subloc_name ON "4014_Episodes_Scene_Locations_Sublocations" USING BTREE(sub_name_y)

--iii
CREATE INDEX source_char_id ON "4014_Character_Relationships"  USING BTREE(source_character_id)
CREATE INDEX rel_type ON "4014_Character_Relationships"  USING BTREE(relationship_type)

--iv
CREATE INDEX h_name ON "4014_Characters"  USING BTREE(housename)
CREATE INDEX target_char_id ON "4014_Character_Relationships"  USING BTREE(target_character_id)


