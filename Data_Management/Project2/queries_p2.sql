


--i
select  ch.charactername,count( distinct sc.id) from 
public."4014_Episodes_Scene_Characters" sc,public."4014_Characters" ch
where sc.char_id=ch.id
group by ch.charactername
order by 2 desc
limit 20

--ii
select distinct ep.id,episodetitle,eplength from public."4014_Episodes" ep, public."4014_Episodes_Scene_Locations_Sublocations" loc
where loc.id=ep.id and sub_name_y='Winterfell'

--iii
select distinct char.id,charactername,housename,actorname0,actorname1 from public."4014_Character_Relationships" rel,public."4014_Characters" char
where rel.source_character_id=char.id and relationship_type='Killed'
order by 2

--iv

select count(*) as Counts from (
select charactername from public."4014_Characters" where housename='Stark'
union
Select  c1.charactername from  public."4014_Characters" c1,public."4014_Character_Relationships" rel,public."4014_Characters" c2
where c1.id=rel.source_character_id and rel.target_character_id=c2.id and  c2.housename='Stark'
and rel.relationship_type in ('Married_engaged','Servant_of') 
)

-- v
UPDATE public."4014_Opening_Locations"
	SET  sub_name='asdf'




