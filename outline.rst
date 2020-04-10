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
* Det kan vi så slippe løs sammen med resten af vores kode, og nu har vi et program, der kan gøre noget, som vi ikke selv har programmet den til
* Det er sådan, at vi bygger machine learning modeller til dagligt - i hvert på et overordnet niveau
* Men nu hedder det her foredrag jo kreative computere, og det at sige om et billede indeholder en bjørn eller en tiger er ikke just kreativt
* For at blive kreative skal vi bruge en generativ model, som er en speciel måde at anvende en machine learning model på
* Inden da tager vi dog lige en fem minutters pause

Del 2
-----
* Nu talte vi i første del om, hvordan man kunne bygge en model, der kunne genkende en tiger
* Samtidig sagde vi, at det ikke var nok til at lave en generativ model
* Faktisk er en generativ model, som ofte skaber noget ud af ingenting, sværere at lave end en der bare skal fortælle, hvad der er på et billede
* Det har simpelthen noget at gøre med, hvor meget den skal tage højde for
* Den model vi lavede i første del skulle kun koncentrere sig om at finde ud af, om der var en løve, en tiger eller en bjørn på billedet
* Den behøvede ikke overveje, hvor realistisk det billede var - Den del var lidt overladt til den, der leverede billedet
* Når vi bygger en generativ model, så skal vi ikke bare bekymre os om, hvad der skal være på billedet, men vi skal også bekymre os om, hvorvidt det er realistisk
* Sagt med andre ord, så skal vi kun skabe ting, som med alt sandsynlighed ville være opstået i virkeligheden
* En model, der genererer grønne tigere eller skaldede løver ville ikke vinde mange point
* Vi venter dog lidt med at kigge på modeller, der skaber billeder
* I den her del kigger vi nemlig på at skabe tekst
* Det er ofte lidt mere tilgængeligt, men samtidig også meget sjovt at computeren generere plausible fortællinger
* Men der er også praktiske anvendelser ved det
* Når vi skaber ny tekst, så er det faktisk de samme teknikker, som chatbots, autocomplete og oversættere som Google Translate anvender
* De er efterhånden så gode, at vi ikke altid opdager, at vi taler med en chatbot i starten, når vi skriver med kundeservice på en hjemmeside
* Her er strategien, at botten starter og når det bliver for svært for den, så overdrages samtalen til en rigtig person
* Det gør nemlig, at der skal færre personer til at bemande sådan en kundeservice, da flere af samtalerne afsluttes før det bliver for svært for botten
* Nå... Men tilbage til vores egne eksempler
* Vi skal jo lige først finde ud af, hvordan en model egentlig kan skabe tekst
* Fordelen med tekst er, at det er en lang streng af ord - det kan vi udnytte
* Så når modellen skal skabe tekst, så ser den på, hvad vi har indtil nu, og så gætter den på hvad det næste ord skal være
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
* For at give en forståelse af det, bygger vi op gennem tre modeller fra simpel til kompliceret
* Den allersimpleste form for hukommelse er, hvor computeren husker de sidste par ord, og så bruger det til at gætte det næste
* Så hvis den er sat til at huske de sidste fire ord, så vi den glemme det første ord, når den læser et nyt
* For at træne modellen skal den bare tælle op, hvilke ord den har set efter alle kombinationer af fire ord og hvor mange gange
* Når det er gjort kan vi bede den om at begynde at skrive tekst
* -- Markov DEMO --
* Nu begynder vi at lægge nye måder ind, hvor modellen kan huske og hvor den skal kigge, men grundideen er altid det samme
* 1. Læse en masse tekst
  2. Finde ud af hvilke ord der kommer efter og hvor ofte
  3. Starte fra et tomt startsted
  4. På skifte læse den tekst der er genereret og indsætte et nyt ord

* Som det næste kan vi lege med hvor langt tilbage i teksten modellen skal kunne huske
* Vi kan skrue på den ved håndkraft, men for lang hukommelse vil betyde, at den bare gentager den tekst, som den oprindeligt læste
* For kort så husker den ikke hvad den selv skrev og laver noget nonsens
* Det er kort sagt noget bøvl selv at skulle finde
* Hvis vi derimod gør det til en del af træningen, at den selv skal lære, hvor meget den skal huske og glemme, så kan vi skabe en smartere model
* Lad os tage et kig på, hvad det er for nogle svagheder, der er i vores model
* For det første vil den altid have en begrænset og unuanceret hukommelse
* Enten husker den et ord eller også glemmer den det - Den har ingen chance for at variere, hvor lang tilbage den skal huske
* Derudover tror den altid, at det sidste ord er det vigtigste - Den har med andre ord ikke mulighed for at søge tilbage i sætningen, ligesom vi mennesker gør det
* Vores store udfordring i at løse det her er at vi arbejder med ord - Det er computere bare ikke særlig gode til, de vil hellere have tal
* Hvis vi 


Del 2
-----
GPT-2 (demo: https://talktotransformer.com/)

Del 3
-----
Neural Style Transfer
GAN
Name drop StyleGAN
DeOldefy
