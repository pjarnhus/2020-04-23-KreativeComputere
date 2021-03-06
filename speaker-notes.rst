Del 1
-----
-- Forside slide
* Velkommen til foredraget om Kreative Computere
* 

-- Tegning af mig + Rambøll logo + Vector med udsnit i skærm, hvor man kan se hjerne
* Mit navn er Philip Jarnhus og jeg arbejder til dagligt hos Rambøll som Data Scientist
* Det betyder, at jeg bruger stort set al min tid på at rode med data og finde måder, at vi kan få computere til at tage beslutninger for os
* Til en stor af det bruger vi machine learning, hvilket også er det vi skal snakke om i dag
* Det er bare ikke altid lige sjovt at høre om, hvordan man bruger machine learning til at optimere projekt styring, så vi tager en lidt anden vej i dag
* Noget af det, der har været meget oppe i tiden er diskussionen om hvorvidt maskiner endelig er blevet intelligente
* Nu siger jeg oppe i tiden, men det er faktisk noget, som folk har snakket om de sidste 70 år
* Og det er hele tiden lige ved at være der
* Det er nok ikke anderledes denne gang, men der er alligevel noget nyt, som vi ikke har set før
* I den her omgang har vi set computere skabe noget, som vi ville kalde kreativt
* Kreativitet, det at skabe noget, er nemlig noget vi ofte anser som et tegn på noget meget menneskeligt og intelligent
* Det betyder ikke, at computere er generelt intelligente, men det er fascinerende at pille ved nogle af de ting, som vi anser for at være unikke for mennesker

-- Tegnet agenda: 1. del = Vector Hugo tænker på en modelkasse, 2. del = Vector Hugo skriver ved skrivebord, 3. del = Vector Hugo maler
* Vi kommer meget til at behandle machine learning modeller som en hyldevare her
* Det vil sige, at vi ikke kigger dybt ned i, hvordan en machine learning model virker, men mere gøre som man gør i virkelige problemer
* Fokusere på hvordan man træner en model og kombinerer modeller til få dem til at virke, som man har brug for
* I første del af foredraget kommer vi til at se på, hvad en model er og hvordan man træner den
* Anden del handler om, hvordan man kan sætte en model op til at skrive tekst
* Tredje og sidste del af foredraget kommer til at handle om, hvordan man skaber billeder
* Det er ofte den del, som de fleste af set eksempler på og som er mest fascinerende
* Mellem hver del kommer vi til at holde en kort pause fem minutter
* Lad os komme i gang

-- Håndsvingskasse tager data ind og spytter resultat ud 
* Der er meget snak om machine learning og det virker som det nye sort, men hvad er det egentlig?
* Det er faktisk ikke andet end et lille stykke software, der tager nogle tal, laver en stak beregninger og kommer frem til et eller flere andre tal
* Hele tricket ligger i, hvordan vi er kommet frem til, hvilke beregninger der skal laves
* For alt software handler egentlig om at tage data ind, ændre det og spytte resultatet ud

-- Traditionelle algoritmer vs. machine learning
* Det er det samme, som I gør hele tiden. I skolen får I f.eks. en opgave, det kan være et regnestykke eller et oplæg til en stil, så behandler I de oplysninger og producerer et resultat
* I mere traditionel programmering skal man selv fortælle computeren, hvordan beregningerne skal se ud
* Det er det, I gør når I skal regne noget. Hvis I skal lægge to tal sammen, så har I lært en fast fremgangsmåde, som I gentager indtil I når frem til et resultat
* Det kaldes en algoritmisk tilgang og er det som en computer gør for det meste af tiden
* Det er rigtig godt i de fleste tilfælde, men det bliver et problem, når beregningerne bliver så svære, at man ikke kan fortælle computeren hvad den rigtige beregning er
* Det er den type problemer, som computere er rigtig dårlige til at løse
* Det er her, at machine learning kommer ind
* I stedet for at fortælle computeren, hvad den skal gøre, så lader vi den lære fra eksempler, hvor vi kender svaret
* I kan lidt sammenligne det med, når I skal skrive en dansk stil
* Der er ikke en standard fremgangsmåde for, hvordan man omdanner et oplæg til en god stil
* Så det man gør som elev er, at man gætter vildt første gang man skal skrive en stil
* Der er ikke nogen af os, der har en chance for at vide, hvordan man skal skrive en stil første gang vi bliver udsat for det
* Heldigvis har vi nogle søde lærer, der retter vores opgave og fortæller os, hvordan vi kan forbedre os
* Det kan være, at vi ikke skal bruge ordet pludselig hele tiden eller at vi skal variere sproget mere til næste gang
* Næste gang vi skal skrive en stil, kan vi så tage de ting med og skrive en lidt bedre stil
* Hele den her proces gentager sig igen og igen over årene, hvor vi bliver bedre og bedre til at skrive stil

-- Vector Hugo --> Stil med karakter --> Vector tænker (tandhjul) --> Loopback
* Det er faktisk det samme, at der sker, når vi træner en machine learning model
* Vi har noget data, hvor vi kender svaret - Lidt ligesom jeres lærer kan se om en stil er god eller ej, så kan vi se, om modellens gæt er rigtig eller ej
* I starten gætter modellen tilfældigt, og så får den en karakter alt efter, hvor tæt på de rigtige svar den ramte
* Modsat jeres karakterer så er modellens karakter beregnet ud fra en ligning, vi kender
* Det betyder også, at modellen vil kunne se, hvordan den skal rette til for at få en bedre karakter i næste forsøg
* Vi fortæller den med andre ord ikke modellen, præcis hvad den gjorde galt, men lader den selv finde ud af det
* Det betyder at vi kan gennemløbe processen rigtig mange gange rigtig hurtigt
* Så modellen gennemgår den samme proces, som I gør når I lærer at skrive en stil
* Laver et gæt -> Får en karakter -> Finder ud af hvordan der skal forbedres -> Laver et nyt gæt
* Fordi vi ikke er inde over hver eneste gang, så kan det gøres rigtig hurtig
* Hvor I er flere år om at lære at skrive en god stil, så kan en model lære at udføre en opgave godt på et sted mellem et par sekunder og en uge
* Skal vi se et eksempel?

** -- DEMO - Løver, Tiger, Bjørne på Teachable Machine -- **

-- Ingen slide - Afslutning på demo siden
* Nu har vi så en model, der har lært at kende løver, tiger og bjørne fra hinanden... I hvert fald næsten
* Hvis den ikke er god nok endnu, så kan vi blive ved med at arbejde med den - give den flere eksempler og justerer hvordan den lærer
* På et tidspunkt er den god nok og så kan vi gemme modellen
* Nu har vi et stykke kode, der indeholder alle de erfaringer, modellen har gjort sig
* Det er med andre ord et stykke kode, der er i stand til at trække de relevante informationer ud fra billedet og omdanne det til et gæt på, hvad der er i billedet
* Det kan vi så slippe løs sammen med resten af vores kode, og nu har vi et program, der kan gøre noget, som vi ikke selv har programmet den til
* Det er sådan, at vi bygger machine learning modeller til dagligt - i hvert på et overordnet niveau
* Men nu hedder det her foredrag jo kreative computere, og det at sige om et billede indeholder en bjørn eller en tiger er ikke just kreativt
* I de næste to dele af foredraget kigger vi på, hvordan vi kan manipulere sådan nogle modeller som dem her til at skabe billeder eller tekst, der fremstår som originale værker
* Inden da tager vi dog lige en fem minutters pause

Del 2
-----
-- Vector ved skrivebord - Tekst generering som titel under ham
* Nu talte vi i første del om, hvordan man kunne bygge en model, der kunne genkende en tiger
* Det er ikke helt nok til at skabe nye og originale værker, men de lægger faktisk grundstenen til, hvordan man kan gøre det
* I den her del kigger vi nemlig på at skabe tekst
* Det er ofte lidt mere tilgængeligt, men samtidig også meget sjovt at computeren generere plausible fortællinger

-- Eksempel billeder: Chatbot, telefon med keyboard fremme, Google Translate
* Vi så i første del modellen tage et billede ind og så gætte på et af tre ord - løve, tiger eller bjørn
* Hvis vi nu ændrer input, så den tager noget tekst ind i stedet for et billede, og så lader den gætte på alle ord i en ordbog, så har vi faktisk muligheden for at lave en model, der kan skabe tekst
* Men der er også praktiske anvendelser ved det
* Når vi skaber ny tekst, så er det faktisk de samme teknikker, som chatbots, autocomplete og oversættere som Google Translate anvender
* De er efterhånden så gode, at vi ikke altid opdager, at vi taler med en chatbot i starten, når vi skriver med kundeservice på en hjemmeside
* Her er strategien, at botten starter og når det bliver for svært for den, så overdrages samtalen til en rigtig person
* Det gør nemlig, at der skal færre personer til at bemande sådan en kundeservice, da flere af samtalerne afsluttes før det bliver for svært for botten

-- En række ord som i en sætning med et spørgsmålstegn i en cirkel til sidst, ord indsat ved spørgsmålstegn i næste linje, nyt spørgsmålstegn efter sætning i tredje linje. Fede pile mellem hvert eksempel
* Nå... Men tilbage til vores egne eksempler
* Vi skal jo lige først finde ud af, hvordan en model egentlig kan skabe tekst
* Fordelen med tekst er, at det er en lang streng af ord - det kan vi udnytte
* Hvis vi har en række ord, så kan vi gætte på det næste ord, sætte det på i halen af den tekst vi allerede har, og så ender vi med en række ord der er et ord længere
* Og det er faktisk kernen i tekstgenerering
  1. Gætter på et ord
  2. Indsætter vores gæt
  3. Starter forfra igen fra punkt 1

-- Vector der kigger opad med spørgsmålstegn over hovedet -> Hangman spil
* Men hvordan gætter den egentlig?
* En computer gætter på bogstaver på samme måde som den gætter på alt andet
* Den lærer hvor sandsynligt hvert tegn er, og så trækker den et tilfældigt ord med den sandsynlighed
* Det lyder måske lidt dumt, men det er faktisk den måde vi andre ofte arbejder på
* Hvis I tænker over det, så er det præcis det, I gør hvis I har spillet hangman
* I starten ved vi intet, så vi gætter på bogstaver som E, R, D og N (Brug småkage som eksempel)
* Det er der en god grund til - Det er nemlig de mest almindelige bogstaver i det danske sprog
* Når der så begynder at blive fyldt bogstaver ind, så gætter vi på bogstaver der passer ind til ord vi kender (T, A, S, I, G, M, K)
* Det er igen samme strategi - Vi har en erfaring om, hvad der er almindeligt - mest sandsynligt - og så gætter vi på det

-- Vector Hugo læser bøger og danner sandsynlighedsfordeling over ord
* Men en computer er ikke så smart som jer - I har selv en idé om, hvad der er det mest sandsynlige at gætte på - En computer skal have lidt mere hjælp
* Vi skal først fortælle den, hvor den skal kigge i sætningen, og så skal den lære hvor sandsynligt hvert ord er
* Det at lære den sandsynligheder er på mange måder det letteste
* Her handler det bare om at lade modellen tygge sig igennem en masse tekst
* Så opbygger modellen en forståelse af, hvad der sandsynligvis passer sammen
* Det er straks sværere at fortælle computeren, hvor den skal kigge i sætningen
* Meget af vores tekstforståelse bygger på, at vi kan huske, hvad der er sket tidligere
* Samtidig ved vi også, hvad forskellige typer af ord bruges til
* Så hvis vi er ved at læse noget, så ved vi, at vi skal kigge efter et navneord, hvis vi vil vide, hvad det er der snakkes om
* Den del skal vi give en computer mulighed for at lære

-- Række af ord, hvor der er kryds over dem, som ikke bliver husket (tre ords hukommelse)
* Den allersimpleste form for hukommelse er, hvor computeren husker de sidste par ord, og så bruger det til at gætte det næste
* Så hvis den er sat til at huske de sidste fire ord, så vi den glemme det første ord, når den læser et nyt
* For at træne modellen skal den bare tælle op, hvilke ord den har set efter alle kombinationer af fire ord og hvor mange gange
* Når det er gjort kan vi bede den om at begynde at skrive tekst

* -- Markov DEMO --

-- Ingen slide - Wrap up i demo vindue
* Det giver faktisk nogle helt fornuftige resultater, hvis man får sat størrelsen af hukommelsen rigtig
* Det minder bare ikke særlig meget om den måde vi mennesker forstår tekst på
* Vi er meget mere komplicerede
* Som mennesker har vi en indbygget evne til at huske ting, som vi har læst langt tilbage i teksten
* Samtidig er vi også i stand til at glemme de ubetydelige dele af en tekst
* Hvis vi bliver spurgt om, hvad det 17. ord i en bog er, så er det de færreste af os, der kan svare
* Men hvis vi bliver spurgt om, hvem der er hovedpersoner, så er det ofte ret let
* Det er fordi, at vi er trænet til at trække de vigtigste oplysninger ud af en tekst gennem hele vores sprogforståelse
* Og vi bruger faktisk rigtig mange forskellige elementer her, som vi kan drage inspiration fra, når vi designer en model

-- "Vi husker det nyeste bedst" stående hen over sliden med en gradient over, så det sidste ord er klart, men det første er næsten forsvundet
* Som det første kan vi huske hvad der er sket gennem det meste af teksten, men vi husker bedst det sidste, der er sket
* Det kommer af, at vi har en forståelse for, at det nyeste er det vigtigste for det, vi skal til at læse
* I computersprog komprimerer vi mentalt hele teksten til en meget lille mængde information, der indeholder de vigtigste oplysninger
* Vi vil ikke kunne genskabe hele teksten ord for ord, men vi har trukket det essentielle ud af teksten og ville kunne skrive en tekst med samme budskab
* Det var på mange måder det, vi lærte modellen i første del
* Den var i stand til at trække det vigtigste information ud af et billede og omforme det til et gæt
* På samme måde kan vi bruge en model til at trække det vigtigste information ud af en tekst

* Det giver bare to nye problemer
  1. Sådan nogle modeller er generelt ikke glade for, når man ikke ved, hvor mange oplysninger der skal håndteres - Billederne var alle en fast størrelse, men vi har ikke nogen måde at vide, hvor lang en tekst er. Det betyder, at vi ikke kan lære modellen, hvor vigtigt det femte sidste ord er. For vi ved ikke, om der er fem ord i sætningen
  2. Det kan være næsten umuligt for modellen at lære på egen hånd, hvad der er vigtigt i sætningen

* Vores redning bliver at gætter på et ord, indsætter det og starter forfra med den nye tekst
* Det gør nemlig, at vi kan snyde lidt
* Når vi første gang har indsat et ord, så har vi jo kun et enkelt nyt ord
* Det betyder samtidig, at alt det hårde arbejde, som vi har gjort for at trække de vigtigste ud af alt den foregående tekst, det har vi allerede gjort
* Hvis vi husker på det fra skridt til skridt, så skal vi bare lære modellen, hvordan den skal opdatere sin viden baseret på et enkelt nyt ord
* Samtidig kan vi også give modellen mulighed for at lære, hvornår den skal glemme noget af det, der tidligere er læst
* Med andre ord kommer vi nærmere og nærmere en god model
* Herefter er det et spørgsmål om at udsætte den for en masse tekst, så den kan lære, hvordan den trækker mening ud af tekst - ligesom da vi viste modellen i første del en masse billeder af tigre, løver og bjørne, så den kunne lære at trække det vigtige viden ud af den del

-- Sætningseksempel med cirkel og pil tilbage til det vigtige ord
* Indtil for et par år siden var det måden at bygge en model på
* Men der mangler stadig en del, som vi mennesker er gode til
* Vi har lært, hvilke dele af en sætning, der er vigtige afhængigt af, hvor vi er
* Det kan vores model her ikke
* Når vi opdaterer vores viden med et nyt ord, glemmer vi med det samme hvordan vores viden så ud før det nye ord
* Det kan nogle gange bringe os i uføre, da modellen kan glemme navne og lignende ting - den ved simpelthen ikke, at den skal kigge tilbage efter det, men forsøger bare at gætte et ord ud fra den totale forståelse af teksten
* Men hvad nu hvis vi gemmer på vores forståelse af teksten som den så ud efter hvert ord?
* Så kan vi tilføje endnu et lag til vores model, hvor den ikke bare lærer at gætte et ord ud fra hvad den kan huske, men samtidig også lærer den hvor i sætningen den skal kigge

-- Håndsvingskasse med stats for model og træning omkring + plus ret linje med formel, hvor parametre er udpeget (gradual reveal)
* Efterhånden har vi ret meget kompliceret model - faktisk er den nu så kompliceret, at de store og mest kraftfulde modeller af slagsen indeholder over 1,5 milliard parametre den kan skrue på
* Hvis du nu lige nu sidder og tænker - hvad er det nu lige en parameter er, så er det de tal, hvor modellen gemmer sin viden
* Det er de tal, der bestemmer hvordan en model opfører sig
* Som eksempel har en ret linje to parametre - hældning og skæringspunkt
* Den her model har mere end 1,5 milliarder
* Det gør også, at den tager temmelig meget data at træne
* Faktisk brugte forskerne teksten fra omkring 8 millioner hjemmesider, som modellen tyggede sig igennem
* Skal vi tage et kig på, hvad den kan?

* -- DEMO : Talk to transformer (https://talktotransformer.com) + Magic Generator (https://minimaxir.com/apps/gpt2-mtg/) --
* Vi tager lige en lille pause mere, og så kigger vi på, hvordan man kan bruge modellerne til at skabe billeder

Del 3
-----
-- Vector som maler med Billedgenerering i tekstboks nedenunder
* Velkommen tilbage til sidste del - nu skal vi til at se på, hvordan man kan skabe billeder
* Der er i virkeligheden flere forskellige teknikker, men vi dykker ned i to styk her til aften

-- Neural Style Transfer slide
* Den første er Neural Style Transfer
* Det er en teknik, hvor man anvender neurale netværk til at overføre stilen af et billede til et andet
* Når vi taler om at overføre stilen fra et billede, så er det farvevalget og måden stregerne i billedet flyder på
* Hvis vi fører det over på et andet billede, så beholder vi indholdet, f.eks. en løve fra det andet billede, men tegner det i samme stil som det første
* Det lyder som lidt af en mundfuld, men faktisk har vi allerede været igennem en stor del af det
* Neurale netværk er en type modeller og det er faktisk dem, vi har brugt til at identificere billeder og også til at generere tekst med den komplicerede af modellerne

-- Flow til at beregne Neural Style Transfer loss
* Det smarte ved de her Neural Style Transfer er, at dem er lavede det indså, at en model der skal identificere et billede, både skal forstå indhold og stil
* Det skal ligesom vide, at noget med striber højst sandsynligt er en tiger, selv om der er mange store kattedyr
* Samtidig skal det også kunne se, at en grøn løve nok ikke er voldsomt realistisk
* For os betyder det, at vi faktisk har gjort det tunge arbejde med modellen, da vi trænede den i første del af foredraget
* Hvis man har sådan en model, så kan vi kigge ind under kølerhjelmen og finde de steder, hvor den koder både stil og indhold
* Det betyder, at vi nu kan starte med at lave gæt på nye billeder, og så se hvordan den klarer sig på de to parametre
* Vi kan med andre ord skabe et nyt billede, tage det billede, plus de to vi skal mikse, og smide igennem vores billedgenkendelsesmodel
* Så får vores gættede billede en samlet karakter for, hvor tæt indhold matcher indholdsbilledet og hvor tæt stil matcher stilbilledet
* I del 1 ville vi så gå tilbage og ændre i modellen for at gøre den bedre til at gætte
* Her ændrer vi i vores gættede billede i stedet
* Teknikken er den samme - Vi får en karakter, ser hvordan vi ændrer det for at forbedre vores karakter og gætter igen
* På den måde kan vi skabe en blanding af to billeder
* Samtidig kan vi skrue på vægten af de to dele i karakteren for at vægte, om det er vigtigst at ramme indhold eller stil
* Lad os lige tage et kig på et par eksempler

* -- DEMO: Neural Style Transfer --

-- Vector Hugo klædt ud som forbryder med maske, og med Deer Stalker og pibe - flow
* Godt så... Vi er kommet en del nærmere at computere opfører sig kreativt, men der er lige en teknik mere, som jeg vil vise jer
* Den hører til blandt mine absolute favoritter og det skyldes især, at måden man træner modellen her på er lidt som at lege politi og røvere med machine learning modeller
* Metoden hedder Generative Adversarial Networks eller GAN
* Det går i alt sin enkelthed ud på, at man har to modeller
* Den ene har rollen som vores forbryder. Dens job er at skabe falsk data, f.eks. falske billeder
* Den anden model er så vores politimand. Den har til opgave at identificere, hvilket noget data der er ægte og hvilket data der er falsk
* I starten er ingen af dem særlig dygtige og det minder mest af alt om en gang Gøg og Gokke
* Forfalsker modellen spytter mere eller mindre bare støj ud, så politimodellen skal ikke være særlig dygtig for at kende forskel
* Men undervejs lærer forfalskeren, hvilke falske data der kan slippe igennem og forbedrer sig
* Det betyder at politimodellen får et hak i trynen for ikke at være god nok
* Men samtidig betyder det også, at den nu har en bedre kvalitet data til rådighed til at træne sig til at kende forskel på ægte og falsk
* Når den så er blevet bedre, så står forfalskermodellen igen med skægget i postkassen, og alt den læring er ikke til megen nytte
* Heldigvis kan den igen lure af, hvad der nu slipper igennem, og så kører det vilde kapløb mellem de to modeller
* Politi modellen bliver bedre til at kende forskel på ægte og falsk, hvilket får forfalskeren til at skabe noget, der minder endnu mere om det ægte data
* Det får så igen politimodellen til at lære at blive bedre til at vide, hvad der er ægte
* Når de to modeller så på et tidspunkt er trænet færdig, kan vi tage forfalsker modellen og bruge den til at skabe nye ting, som ikke har eksisteret før

-- Style GAN ansigter
* Det kan man lave mange sjove ting ud af
* Man kan blandt andet skabe nye ansigter baseret på billeder af kendte som her - ingen af de her mennesker eksisterer i virkeligheden

-- Vector ser billede med vej, træer og huse, men tankeboblen viser kun vejen
* Det kan også bruges til noget mere praktisk
* F.eks. kan man bruge det til at fjerne de elementer fra et kamerabillede, som ikke er relevante
* Hvis man nu vil træne en bil til at køre selv, men kun har mulighed for at træne den i en simulation som MarioKart, så kan man tage nogle billeder fra virkeligheden og bedre forfalsker modellen omdanne dem så det ligner noget for MarioKart
* På den måde får man en nem og billig måde til at træne en model til at køre bil, som ikke bliver forvirret, når den kommer ud og ser alle husene og reklamerne i virkeligheden
* Det kræver bare at man har optaget noget video på et kamera, så forfalskeren har noget input data, men ellers kan man klare det hele på sin computer, hvilket er en del billigere
* Inden I går ud og forsøger det her, så tænk lige på, at noget af det modellen lærer at sortere fra er cyklister og fodgængere
* De eksisterer ikke i MarioKart, men det er ret vigtigt ikke at ramme dem ude i trafikken, så der skal noget ekstra til for at håndtere dem

-- Slide med ordet DeOldify
* Her til sidst vil jeg lige vise jer et andet eksempel fra et projekt, der hedder DeOldify
* De har trænet en model til at farvelægge sort/hvid billeder
* Her bruger de forfalsker modellen til at farvelægge et sort/hvid billede og så skal politimodellen kende forskel på et farvelagt billede og et der oprindeligt var i farver
* Lad mig lige vise jer et eksempel på, hvordan det virker

-- DEMO: DEOldify --

* Det var en så det sidste eksempel i dag
* Jeg håber, at det har givet en introduktion til, hvordan vi kan få computere til at skabe tekst og billeder
* Ofte er det sværeste i det at lure, hvordan man kan anvende en model på en sjov måde
* Det er faktisk ofte udfordringen, når vi arbejder med machine learning
* Som I har set, så kan modellerne godt skabe nye ting, men de bliver trænet til at skabe en type data
* Den samme model kan ikke både skabe tekst og billeder, med mindre den er bygget specifikt til det
* Samtidig er modellerne også bundet af det data, de er trænet på
* Hvis I træner en tekstmodel udelukkende på H.C. Andersens eventyr, så lærer den aldrig nogensinde at skrive i samme sprog som vi bruger på sociale medier
* Den tror alle tekster starter med Der var engang...
* Det er på samme måder med billedmodellerne
* De kender kun de ting, de er trænet på og for dem er det alt hvad der er i verden
* I har måske kunnet fornemme, at det faktisk ikke er så svært at lege med den slags modeller
* Så længe at vi ikke skal bygge dem selv, så kan man hurtigt komme i gang med at lege med dem og få nogle gode resultater
* De her slides bliver lagt op på UNFs hjemmeside og på sidste slide kan I finde links til alle de ting, jeg viste i aften - Så kan I bare tage fat og lege, og forhåbentlig skabe noget fuldstændig fantastisk
* Her er der ikke andet tilbage end at sige tak for i aften
* Tak fordi I var med på den her lidt alternative foredragsform
* Jeg håber, at I har nydt det
* Hvis I har nogle spørgsmål, så skriv dem i kommentarfeltet. Jeg bliver hængende her, så der er rigeligt tid til at få skrevet noget
