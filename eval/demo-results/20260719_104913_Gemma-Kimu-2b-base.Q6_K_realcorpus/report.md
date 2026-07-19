# Real-Corpus Autocomplete Eval — Gemma-Kimu-2b-base.Q6_K.gguf

- **Date:** 2026-07-19
- **Model:** `Gemma-Kimu-2b-base.Q6_K.gguf`
- **Mode:** word-COMPLETION (no trailing space)
- **Output:** filtered (deployed defaults)
- **Tokens:** 8
- **Prompts:** 40 (Wikipedia + Berria)
- **digit artifacts:** 3/40 (8%)
- **first-word exact match:** 5/40 (12%)
- **any overlap:** 11/40 (28%)
- **prefix on track:** 1/40 (2%)
- **avg confidence:** 0.421

## Results (prompt + MODEL + [gold])

| ✓ | Prompt + suggestion | Gold | Conf |
|---|---|---|---|
| ~ | `Horri lotuta, Prozedura`** Zibilaren Legea aldat** | Kriminalaren Legea aldatzea itundu dute. | 0.71 |
|   | `Jendea harat`**ago joaten da, eta hori** | eta honat zebilen ihesi. | 0.33 |
|   | `Oraingo ikuspegitik, ezingo litzateke euskaraz bizi,`** ezta? Ez. Gaur** | euskara baturik gabe. | 0.28 |
|   | `Gipuzkoako ipar-ekialdeko`** herrietan, 201** | euskalkia, aldiz, goi-nafarrera da . | 0.32 |
|   | `Estatua da`**, baina ez du estatuaren funt** | hezkuntza sistemaren gestio eta erregulazioaren erantzule. | 0.14 |
|   | `Patronalen jarrera`** salatu dute sindikatue** | ere gaitzetsi du Aranburuk. | 0.69 |
|   | `LHKk langile`**ek enpresen aurrean d** | espainiarrak ez ditu atzerritartzat hartzen. | 0.42 |
|   | `Eraberritzeko proiektuak`** 2015eko martxo** | dozenaka dira Bilbo guztian. | 0.33 |
|   | `Batasuna giputz euskaran zergatik oinarritu zen azaltzeko, arrazoi`** batzuk eman ditu. Eus** | demografikoak ematen dira gehienetan. | 0.23 |
|   | `Erakusketarako sortu duen testua artxiboaren garrantziaz mintzatzen da,`** eta horren inguruan hausn** | hain zuzen ere. | 0.34 |
|   | `Halaber, aditzera eman du ezen, itzulpena ezarri beharko balitz,`** ez litzatekeela izango hiz** | aurrekontu handiagoa beharko lukeela LHKk. | 0.39 |
| ~ | `Malda txikia amildegi`** handi bat da. {\an2** | itxurakoa da, eta prezipitazioak gero eta urriagoak dira. | 0.33 |
| ~ | `Musikaren sorkuntza, interpretazioa, esanahia eta definizioa ere aldatu egiten dira kulturaren eta`** gizartearen arabera.** | testuinguru sozialaren arabera. | 0.67 |
|   | `Hezkuntza Ministerioaren edo antzeko organismoen bitartez, gobernatzaileak, lurralde`**-administrazioko organoen bidez** | bakoitzeko hezkuntzaren garapenerako oinarriak finkatzen dituzte. | 0.36 |
|   | `Horrez gain, Zinemateka delako programaren barnean garai ezberdinetako`** filmak eskainiko dituzte.** | zine emanaldiak eskaintzen dira. | 0.59 |
|   | `Oroitarritik zentimetro gutxira aparkatzen zuten ibilgailuek,`** eta horrek eragiten zuen traf** | hura ukitzeko moduan ia. | 0.28 |
|   | `Honako hauek dira: Aramitze, Arriba, Arribalda, Azpilda, Ereta,`** Etxeberria, Goikoet** | Hagoeta, Jeruntze, Lixoze, Nabarzi, Orin eta Tabaila-Uzkaine. | 0.39 |
|   | `Erdi Aroan gaztelaniaz idatzi zuen Gonzalo de Berceo Errioxako`** poetak. Gonzalo de Ber** | idazlea ziur aski euskalduna zen. | 0.40 |
|   | `Langabeziari dagokionez, Euskal Autonomi Erkidegoko hiriburuetan tasa`** altuenak dituzten udalerriak dira** | baxuena dago Donostian. | 0.51 |
| ✓ | `Hainbat ibaiadar ditu: Leitzaran, Berastegi,`** Amezketa eta Araxes.** | Amezketa, Araxes, Amundarain, Agauntza eta Urtsuaran. | 0.46 |
| ~ | `Segurtasun Batzordea ere sei hilabete`**an behin bilduko da,** | barru bilduko da, adostutakoaren jarraipena egiteko. | 0.71 |
|   | `Hirietako hainbat euskaldunek, euskalkien erreferentzia sendorik`** ez dutenek ere bai, es** | gabe, euskara batua ama-hizkuntzatzat ikasi du. | 0.29 |
| ✓ | `Askotan gogorragoa egiten zaio artxiboan aukeraketa egitea, horrekin`** lan egitea baino.** | lan egitea baino. | 0.63 |
|   | `Hirukotea osatzeko lehen`**engo pausoa eman dute.** | porrotaren ondoren, PSN-ko hainbat kideren kritika zorrotzak egon ziren. | 0.45 |
|   | `Ming dinastiaren garaian hasi zen jai eta ospakizunetako musika kodetzen`**. Garai hartan, txin** | eta kontserbatorioetan irakasten. | 0.38 |
|   | `Erdi Aroko Europa kristauko unibertsitate haiek oso`** ezberdinak ziren, eta hor** | ongi errotuta zeuden Mendebaldeko Europan. | 0.41 |
| ~ | `Euriak udazkenean eta udaberrian ugariak izaten dira; negua, berriz, eztia,`** lorez betea. Uda** | eta uda ez oso beroa. | 0.30 |
|   | `Plazaren ingurumarietan korrika zebiltzanek gomazko pilotak ez,`** baina bestelakoak jaso** | su armen hotsak aditu zituzten. | 0.43 |
| ✓ | `Artistak adibidetzat jarri du nola argazki guztiak batera jartzean erlazio bat sortzen den eta elkarrekin beste`** zerbait bihurtzen diren. «** | zerbait sortzen duten. | 0.60 |
| ✓ | `Tokiko irrati kate txikiagoak ere badira, adibidez, Bizkaia Irratia eta Bilbo`** Hiria Irratia. Euskal** | Hiria Irratia, euskarazkoak eta Bilbotik herrialde osorako igortzen dutenak. | 0.60 |
|   | `Ertzaintzak Europako datu baseetarako sarbidea izatea ere hitzartu dute, baita`** «terrorismoaren» aurkako** | segurtasun pribatuaren eta zibersegurtasunaren arloetan gaitasun handiagoa izatea ere. | 0.43 |
| ~ | `Europarrak heldu aurretik indiarrek ez zuten ezagutzen musika notaziorako sistemarik, musika tresnek laguntzen zioten kantuari, eta musika eta dantza`** erabiltzen ziren errituetan** | batera egiten ziren beti. | 0.26 |
|   | `Azken urteotan, hizkuntza batua sortzeko ekimenak ezinbestekotzat jo izan dituzte aragoieraren aldeko`**ek. Aragoiera ez da** | kulturgileek, mintzairaren erabilera behin betiko itzal ez dadin. | 0.38 |
|   | `Hori bai, LHKk zehaztu du gaur egun ez duela itzulpen zerbitzurik, eta ez dagokiola berari`** hori eskaintzea.** | erabakitzea bileretan halakorik erabiliko den ala ez. | 0.50 |
| ✓ | `Haren ustez, Confebaskek eta CENek uko egin diote beren erantzukizunari, eta`** «ez dute nahi» negozia** | ez dute borondaterik sektore arteko akordio bat negoziatzeko. | 0.25 |
|   | `Aipatzekoak dira, halaber, hiriko auzoetan zehar ospatzen diren`** jai eta ospakizunak. Hor** | jaiak, hala nola Amara Zaharreko jaiak eta Egiako Porrontxoak. | 0.33 |
|   | `Morfologikoki hizkuntza eranskaria delako ezaugarria azpimarratu izan da`**. Hala ere, euskararen** | euskararako, deklinabidea monemak banan-banan erantsita sortu dela uste izanda. | 0.49 |
|   | `Gerra horien ondoren sortutako ezinegonetik abiatu ziren bai kontzertu ekonomikoaren ideia, eta`** baita nazioarteko harre** | foruen galeratik abiatu zen ere euskal nazionalismoa. | 0.50 |
|   | `Eremu biologiko, nutrizionak, familiar edo ingurugiroan sortutako desberdintasun fisikoak, psikikoak eta sozialak; programa berezien eta komunitateko beste erakunde batzuekin artikulatutako ekintzen bitartez, hauek`** konpontzeko helburuarekin.** | prebenitu eta artatu. | 0.47 |
|   | `Beste hainbeste jaik, berriz, inauterietan zentratzen dira eta oso famatuak dira, adibidez,`** Tolosakoa. Inauteri** | Ituren eta Zubietan ospatzen direnak, baita Sakana aldean ere. | 0.30 |