# Blinded A/B Evaluation — Basque Autocomplete

**20 prompts** from held-out Basque text (Wikipedia, never seen in training).
Each prompt has two completions: **A** and **B**. One is from an earlier checkpoint,
one from a later checkpoint — randomly assigned per prompt.

## Instructions for the expert

For each pair, judge which completion is **better Basque** — more grammatical,
more natural, more useful as an autocomplete suggestion.

- **A** = completion A is better
- **B** = completion B is better
- **T** = tie (both equally good, or both equally bad)

Judge on **Basque quality**, NOT on whether it matches the original text.
Both completions may differ from the original but still be valid Basque.
If both are garbage, mark **T**.

Record your judgments as a list, e.g.: `1:A 2:B 3:T 4:A ...`

---

### 1
**Prompt:** `Atzizki horrek, zenbait egileren`
**A:** `arabera, 1990eko hamarkadan 1990`
**B:** `arabera, kokagunea adierazten du, eta, beraz, `
**Your judgment:** ___

### 2
**Prompt:** `Gainera, plazaren konposaketa`
**A:** `eta antolamendua ere aldatu egin da.`
**B:** `eta kokapena ere aldatu egin da.`
**Your judgment:** ___

### 3
**Prompt:** `Merkatu berri horretan, monopolioen presentzia`
**A:** `k eta komertzioen komertzioen `
**B:** `k eta komertzializazioak eragin handia izan dute.`
**Your judgment:** ___

### 4
**Prompt:** `Masami Kurumadak, hasiera batean, zalantza egin zuen Geminisko zalduna izaera bikoitzeko`
**A:** `izaki bat zela, eta, beraz, Geminisko za`
**B:** `pertsonaia gisa deskribatzen zuen, baina, azkenean, Gemini`
**Your judgment:** ___

### 5
**Prompt:** `Oso funtzio garrantzitsua betetzen du bandetan,`
**A:** `eta, beraz, oso garrantzitsua da taldeen arteko harremanak sendotzea.`
**B:** `eta, beraz, zaleek ere parte har dezakete.`
**Your judgment:** ___

### 6
**Prompt:** `Eta esaten zaien egunetan`
**A:** `, 100.000 biztanleko 100`
**B:** `, 100.000 biztanleko 100`
**Your judgment:** ___

### 7
**Prompt:** `Komorbilitateari, haren bilakaerari eta aldi berean gaixotasun ugari`
**A:** `ri aurre egiteko.`
**B:** `ri aurre egiteko.`
**Your judgment:** ___

### 8
**Prompt:** `Bizitza arriskuan duela adieraziko dion bizilagunetik hasi eta etxean hartuko duen prostitutaraino, sostengatuko duen`
**A:** `bizilagunak bizilagunaren bizi-kalitatea hobetuko du.`
**B:** `bizilaguna izango da.`
**Your judgment:** ___

### 9
**Prompt:** `Material ugari landu`
**A:** `da, eta, beraz, ez da erraza izan.`
**B:** `ko da, eta, horretarako, hainbat material eta teknika erabiliko dira.`
**Your judgment:** ___

### 10
**Prompt:** `Erromatartze garaian, Junonekin parekatu zuten, eta ez Venusekin, Astarterekin parekatuz gero`
**A:** `.`
**B:** `.`
**Your judgment:** ___

### 11
**Prompt:** `Akademiko gisa, La Pazko San Andres Unibertsitate Nagusiako eta`
**A:** `La Pazkoko Unibertsitateko irakaslea izan zen.`
**B:** `La Pazkoko Unibertsitateko irakaslea izan zen.`
**Your judgment:** ___

### 12
**Prompt:** `Album hori NGSk argitaratu`
**A:** `zuen 2016an, eta 2017an`
**B:** `zuen 2013an, eta 2014an`
**Your judgment:** ___

### 13
**Prompt:** `Hasiera batean lur horien kolektibizazioa lortu nahi zen, eta honekin batera bertako lurralde batzuk`
**A:** `, hala nola, kolektibitatearen kolektibitate`
**B:** `, hala nola, Aragoi, Gaztela eta Leon, Gazte`
**Your judgment:** ___

### 14
**Prompt:** `Borroketan hildako asko izateaz gain, frantziar soldaduen artean izurrite bubonikoa agertu zen, eta`
**A:** `, ondorioz, 1939ko uztailean, Frantziako`
**B:** `1940ko hamarkadan 1.000 hil`
**Your judgment:** ___

### 15
**Prompt:** `Jugoslaviako Errepublika Federala osatzen zuten sei errepubliken artean Mazedonia zen`
**A:** `lehena.`
**B:** `1914an.`
**Your judgment:** ___

### 16
**Prompt:** `Zis-Golgi zonaldea nukleo eta erretikulu endoplasmatikotik gertuen`
**A:** `dagoen gunea da.`
**B:** `dagoena da.`
**Your judgment:** ___

### 17
**Prompt:** `Parte hartzen duen errebonbillo bakoitzak neska`
**A:** `bat izango du, eta mutil bat, neska bat.`
**B:** `bat du buruan.`
**Your judgment:** ___

### 18
**Prompt:** `Ikergazte kongresua eta`
**A:** `2019ko urtarrilean EHUko Udako Ikas`
**B:** `2019ko edizioa`
**Your judgment:** ___

### 19
**Prompt:** `José Lopesen biloba`
**A:** `, 1936ko uztailaren 18an jaioa.`
**B:** `, 1936ko irailaren 11n hil zen,`
**Your judgment:** ___

### 20
**Prompt:** `Pradojikoki, komunikabideen influentzia komunikazio`
**A:** `an.`
**B:** `aren kodeen eta kodeen kodeen artean`
**Your judgment:** ___
