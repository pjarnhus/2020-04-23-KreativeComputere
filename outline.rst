Del 1
-----
* Velkommen til foredraget om Kreative Computere
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
* Vi kommer meget til at behandle machine learning modeller som en hyldevare her
* Det vil sige, at vi ikke kigger dybt ned i, hvordan en machine learning model virker, men mere gøre som man gør i virkelige problemer
* Fokusere på hvordan man træner en model og kombinerer modeller til få dem til at virke, som man har brug for
* I første del af foredraget kommer vi til at se på, hvad en model er og hvordan man træner den
* Anden del handler om, hvordan man kan sætte en model op til at skrive tekst
* Tredje og sidste del af foredraget kommer til at handle om, hvordan man skaber billeder
* Det er ofte den del, som de fleste af set eksempler på og som er mest fascinerende
* Mellem hver del kommer vi til at holde en kort pause fem minutter
* Lad os komme i gang
* Der er meget snak om machine learning og det virker som det nye sort, men hvad er det egentlig?
* Det er faktisk ikke andet end et lille stykke software, der tager nogle tal, laver en stak beregninger og kommer frem til et eller flere andre tal
* Hele tricket ligger i, hvordan vi er kommet frem til, hvilke beregninger der skal laves
* For alt software handler egentlig om at tage data ind, ændre det og spytte resultatet ud
* Det er det samme, som I gør hele tiden. I skolen får I f.eks. en opgave, det kan være et regnestykke eller et oplæg til en stil, så behandler I de oplysninger og producerer et resultat
* I mere traditionel programmering skal man selv fortælle computeren, hvordan beregningerne skal se ud
* Det er det, I gør når I skal regne noget. Hvis I skal lægge to tal sammen, så har I lært en fast fremgangsmåde, som I gentager indtil I når frem til et resultat
* Det kaldes en algoritmisk tilgang og er det som en computer gør for det meste af tiden
* Det er rigtig godt i de fleste tilfælde, men det bliver et problem, når beregningerne bliver så svære, at man ikke kan fortælle computeren hvad den rigtige beregning er
* I kan lidt sammenligne det med, når I skal skrive en dansk stil
* Der er ikke en standard fremgangsmåde for, hvordan man omdanner et oplæg til en god stil
* Så det man gør som elev er, at man gætter vildt første gang man skal skrive en stil
* Der er ikke nogen af os, der har en chance for at vide, hvordan man skal skrive en stil første gang vi bliver udsat for det
* Heldigvis har vi nogle søde lærer, der retter vores opgave og fortæller os, hvordan vi kan forbedre os
* Det kan være, at vi ikke skal bruge ordet pludselig hele tiden eller at vi skal variere sproget mere til næste gang
* Næste gang vi skal skrive en stil, kan vi så tage de ting med og skrive en lidt bedre stil
* Hele den her proces gentager sig igen og igen over årene, hvor vi bliver bedre og bedre til at skrive stil
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
* Nu talte vi i første del om, hvordan man kunne bygge en model, der kunne genkende en tiger
* Det er ikke helt nok til at skabe nye og originale værker, men de lægger faktisk grundstenen til, hvordan man kan gøre det
* I den her del kigger vi nemlig på at skabe tekst
* Det er ofte lidt mere tilgængeligt, men samtidig også meget sjovt at computeren generere plausible fortællinger
* Vi så i første del modellen tage et billede ind og så gætte på et af tre ord - løve, tiger eller bjørn
* Hvis vi nu ændrer input, så den tager noget tekst ind i stedet for et billede, og så lader den gætte på alle ord i en ordbog, så har vi faktisk muligheden for at lave en model, der kan skabe tekst
* Men der er også praktiske anvendelser ved det
* Når vi skaber ny tekst, så er det faktisk de samme teknikker, som chatbots, autocomplete og oversættere som Google Translate anvender
* De er efterhånden så gode, at vi ikke altid opdager, at vi taler med en chatbot i starten, når vi skriver med kundeservice på en hjemmeside
* Her er strategien, at botten starter og når det bliver for svært for den, så overdrages samtalen til en rigtig person
* Det gør nemlig, at der skal færre personer til at bemande sådan en kundeservice, da flere af samtalerne afsluttes før det bliver for svært for botten
* Nå... Men tilbage til vores egne eksempler
* Vi skal jo lige først finde ud af, hvordan en model egentlig kan skabe tekst
* Fordelen med tekst er, at det er en lang streng af ord - det kan vi udnytte
* Hvis vi har en række ord, så kan vi gætte på det næste ord, sætte det på i halen af den tekst vi allerede har, og så ender vi med en række ord der er et ord længere
* Og det er faktisk kernen i tekstgenerering
  0. Gætter på et ord
  0. Indsætter vores gæt
  0. Starter forfra igen fra punkt 1
* Men hvordan gætter den egentlig?
* En computer gætter på bogstaver på samme måde som den gætter på alt andet
* Den lærer hvor sandsynligt hvert tegn er, og så trækker den et tilfældigt ord med den sandsynlighed
* Det lyder måske lidt dumt, men det er faktisk den måde vi andre ofte arbejder på
* Hvis I tænker over det, så er det præcis det, I gør hvis I har spillet hangman
* I starten ved vi intet, så vi gætter på bogstaver som E, R, D og N (Brug småkage som eksempel)
* Det er der en god grund til - Det er nemlig de mest almindelige bogstaver i det danske sprog
* Når der så begynder at blive fyldt bogstaver ind, så gætter vi på bogstaver der passer ind til ord vi kender (T, A, S, I, G, M, K)
* Det er igen samme strategi - Vi har en erfaring om, hvad der er almindeligt - mest sandsynligt - og så gætter vi på det
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
* Den allersimpleste form for hukommelse er, hvor computeren husker de sidste par ord, og så bruger det til at gætte det næste
* Så hvis den er sat til at huske de sidste fire ord, så vi den glemme det første ord, når den læser et nyt
* For at træne modellen skal den bare tælle op, hvilke ord den har set efter alle kombinationer af fire ord og hvor mange gange
* Når det er gjort kan vi bede den om at begynde at skrive tekst
* -- Markov DEMO --
* Det giver faktisk nogle helt fornuftige resultater, hvis man får sat størrelsen af hukommelsen rigtig
* Det minder bare ikke særlig meget om den måde vi mennesker forstår tekst på
* Vi er meget mere komplicerede
* Som mennesker har vi en indbygget evne til at huske ting, som vi har læst langt tilbage i teksten
* Samtidig er vi også i stand til at glemme de ubetydelige dele af en tekst
* Hvis vi bliver spurgt om, hvad det 17. ord i en bog er, så er det de færreste af os, der kan svare
* Men hvis vi bliver spurgt om, hvem der er hovedpersoner, så er det ofte ret let
* Det er fordi, at vi er trænet til at trække de vigtigste oplysninger ud af en tekst gennem hele vores sprogforståelse
* Og vi bruger faktisk rigtig mange forskellige elementer her, som vi kan drage inspiration fra, når vi designer en model
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
* Indtil for et par år siden var det måden at bygge en model på
* Men der mangler stadig en del, som vi mennesker er gode til
* Vi har lært, hvilke dele af en sætning, der er vigtige afhængigt af, hvor vi er
* Det kan vores model her ikke
* Når vi opdaterer vores viden med et nyt ord, glemmer vi med det samme hvordan vores viden så ud før det nye ord
* Det kan nogle gange bringe os i uføre, da modellen kan glemme navne og lignende ting - den ved simpelthen ikke, at den skal kigge tilbage efter det, men forsøger bare at gætte et ord ud fra den totale forståelse af teksten
* Men hvad nu hvis vi gemmer på vores forståelse af teksten som den så ud efter hvert ord?
* Så kan vi tilføje endnu et lag til vores model, hvor den ikke bare lærer at gætte et ord ud fra hvad den kan huske, men samtidig også lærer den hvor i sætningen den skal kigge
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
Neural Style Transfer
GAN
Name drop StyleGAN
DeOldefy
