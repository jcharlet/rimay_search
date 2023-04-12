from ast import List
import streamlit as st
from src.models.predict_model import (
    run_query_with_qa_with_sources,
    COL_STATE_OF_THE_UNION,
    COL_OPEN_MINDFULNESS,
    ResponseSize,
)


# Define function to run query and display results
def run_query(
    query,
    mocked=False,
    collection_name=COL_OPEN_MINDFULNESS,
    response_size=ResponseSize.MEDIUM,
):
    if mocked:
        answer = """The president said "Justice Stephen Breyer—an Army veteran, Constitutional scholar, and retiring Justice of the United States Supreme Court. Justice Breyer, thank you for your service." """
        # sources_json = [{'id': '31-pl', 'document': 'Tonight. I call on the Senate to: Pass the Freedom to Vote Act. Pass the John Lewis Voting Rights Act. And while you’re at it, pass the Disclose Act so Americans can know who is funding our elections. \n\nTonight, I’d like to honor someone who has dedicated his life to serve this country: Justice Stephen Breyer—an Army veteran, Constitutional scholar, and retiring Justice of the United States Supreme Court. Justice Breyer, thank you for your service. \n\nOne of the most serious constitutional responsibilities a President has is nominating someone to serve on the United States Supreme Court. \n\nAnd I did that 4 days ago, when I nominated Circuit Court of Appeals Judge Ketanji Brown Jackson. One of our nation’s top legal minds, who will continue Justice Breyer’s legacy of excellence.', 'metadata': {'source': '31-pl'}}]
        sources_json = [
            {
                "id": "3.5.20.01",
                "document": "6. L’intégration des émotions avec la méthode en trois temps\n\nLes trois méthodes d’intégration des pensées (dans l’ouverture relâchée, dans l’observateur abstrait, avec et sans étiquetage) sont susceptibles d’être transposées à l’intégration des émotions, qui sont, comme nous l’avons dit, des pensées « énergétisées », investies d’une charge affective. Vous extrapolerez donc ces trois méthodes, l’approche est exactement la même.\nL’entrainement progressif consiste à commencer à s’entrainer en situation facile avec les pensées habituelles. Puis, lorsque l’on s’est habitué à intégrer les pensées habituelles il devient plus facile d’intégrer d’abord des petites émotions, puis progressivement de fortes émotions et finalement toutes les émotions.\nMaintenant nous allons voir une pratique spécifique d’intégration et de libération des émotions dans la pleine présence. Elle a trois étapes :\n\nLa première étape consiste à reconnaître et accueillir la présence de l’émotion.\nLa deuxième étape consiste à respirer avec l’émotion. En respirant nous incorporons l’émotion et la laissons se décharger dans l’ouvert.\nLa troisième étape consiste, apaisé après la décharge de l’émotion, à rester en l’état d’observateur abstrait en harmonie avec la situation.\n\n Nous allons développer et expérimenter ces trois étapes :\nPremière étape : la reconnaissance et l’accueil\nDescriptif\nLe propos de cet exercice est de reconnaitre la présence de l’émotion. Habituellement, nous ne repérons pas l’arrivée d’une émotion. Si nous pouvions voir l’émotion dès qu’elle apparaît, nous pourrions ne pas nous laisser emporter par son énergie. Pour ce faire, « le tableau de bord corporel » présenté dans la première étape est très utile. Ce que nous avons alors nommé « le baromètre corporel » nous permet de déceler l’émergence d’une émotion à partir de nos sensations corporelles, en étant attentifs à la sensation de notre corps.\nExercice (Durée : 5 mn)\n\nComme d’habitude : installez-vous dans une posture confortable, les sens tout ouverts, détendus dans le regard panoramique, ça respire.\n……\n\n\nDans cet état : sentez votre corps, observez-le. Observez ce qui s’y passe et déceler y les signes d’émergence d’une émotion.\n……\n\n\nNotez particulièrement comment est votre respiration, comment bat votre cœur, les petites sensations dans votre ventre, gorge ou ailleurs, traduisant la présence d’une émotion. Cette attention à la sensation du corps vous aide à reconnaitre l’apparition d’une émotion.\n……\n\n\nSi vous n’avez dans l’instant aucune émotion décelable vous pouvez évoquer une situation émotionnelle de votre vie récente. Vous trouverez certainement facilement une situation qui vous a provoqué une émotion. Evoquez-la et sentez l’émotion.\n……\n\n\nA présent ayant reconnu la présence d’une émotion, vous l’accueillez. Plutôt que de vouloir la suivre ou la fuir, vous accueillez et acceptez simplement sa présence, sans jugement. Vous sentez pleinement son énergie, sa chaleur : cela peut se manifester de différentes façons suivant la nature de l’émotion. Se peut être « la moutarde qui monte au nez », une production accrue d’adrénaline, une respiration plus intense, un rythme cardiaque accéléré… Quoi qu’il en soit, vous accueillez simplement cette émotion et son énergie avec bienveillance, dans une sorte de « oui » qui est comme un sourire bienveillant, un accueil sans jugement.\n……\n\n\nNe vous demandez pas s’il est bien que cette émotion soit là ou pas. Restez simplement dans un état d’attention relâché, accueillant l’émotion, la laissant venir.\n……\nSi cela vous aide, vous pouvez comme pour les pensées avoir recours à la reconnaissance verbale avec l’étiquetage : « émotion » ou encore à la simple reconnaissance non verbale. Cette reconnaissance permet un accueil neutre et bienveillant des émotions, permet d’entrer en amitié avec ce que l’on porte d’émotionnel et de conflictuel en soi.\n……\n\nDeuxième étape : Respirer dans l’émotion, l’incorporant et la laissant se décharger\nDescriptif\nLa deuxième étape consiste à respirer dans l’émotion en l’incorporant et la laissant se décharger dans l’ouvert. Ayant accueilli et accepté la présence de l’émotion nous utilisons l’alternance du cycle respiratoire pour incorporer et laisser se décharger l’émotion. Particulièrement dans l’inspiration nous accueillons son énergie et l’incorporons, faisons corps avec elle. Dans l’expiration nous nous relâchons complètement et laissons cette énergie se décharger dans l’ouvert.\nExercice (Durée : 5 mn)\n\nToujours installé dans la posture confortable, les sens tout ouverts et détendu dans le regard panoramique, ça respire naturellement.\n……\n\n\nA présent, après avoir accueilli l’émotion, vous vous familiarisez avec sa présence et entrez en son énergie sans peur ni résistance. Vous restez dans son ressenti en respirant dans sa sensation.\n……\n\n\nVous incorporez l’émotion, vous vous laissez aller dans la sensation de son énergie, faisant de plus en plus corps avec elle.\n……\n\n\nCette incorporation de l’émotion se vit associée à la respiration, à la pulsation du souffle. Vous vous laissez aller dans celle-ci au rythme de l’inspiration et de l’expiration. Particulièrement, vous associez l’accueil avec l’inspiration et le lâcher prise dans l’ouvert avec l’expiration.\n……\n\n\nEn respirant ainsi, vous vous détendez et vous vous abandonnez dans l’énergie de l’émotion. Dans le va-et-vient de la respiration, progressivement, vous incorporez l’émotion et la laissez se décharger.\n……\n\n\nVous continuez ainsi laissant l’énergie de l’émotion être telle qu’elle est. Dans le lâcher prise associé à la détente la distance entre vous et l’émotion se réduit et finalement disparaît, laissant simplement son énergie dans l’ouvert. L’émotion devient alors une énergie qui n’est pas possédée, une énergie libre qui rayonne et se décharge.\n……\n\n\nEn pratiquant ainsi le temps nécessaire, l’intensité émotionnelle diminue, retombe et finalement se dissout. Le reliquat de son énergie n’est plus conflictuel et peut même devenir source d’une intelligence qui va animer la troisième étape.\n……\n\nTroisième étape : Rester apaisé dans l’état d’observateur abstrait en harmonie avec la situation\nDescriptif\nDans cette troisième étape, une fois que l’émotion s’est déchargée nous  restons tel quel dans la simple présence ouverte globale, en état d’observateur abstrait. Nous sommes alors en harmonie avec la situation et pouvons, si la situation le demande, y répondre de façon adaptée.\n Exercice (Durée : 5 à 10 mn)\n\nToujours installé confortablement, les sens tout ouverts et détendu dans le regard panoramique, ça respire.\n……\n\n\nPour cette troisième étape d’intégration de l’émotion, vous restez apaisé dans l’ouvert, en l’état d’observateur abstrait. Lors de la deuxième étape la charge émotionnelle s’est déchargée, le corps et l’esprit sont maintenant relâchés. Vous restez au repos, tel quel, dans la simple présence ouverte de l’observateur abstrait.\n……\n\n\nNaturellement et tranquillement, ça respire. Vous êtes un, apaisé dans la situation.\n……\n\n\nDans cet état de pleine présence, attentive, ouverte et empathique, vous êtes en harmonie avec la situation et son intelligence, et ainsi, une réponse adaptée peut émerger, que ce soit de dire ou faire quelque chose ou de ne rien dire ni rien faire.\n……\n\n\nAu départ, quand l’émotion était virulente, vous auriez réagi émotionnellement. Mais à présent, l’émotion s’étant déchargée, vous pouvez répondre de façon harmonieuse et empathique, en percevant la situation sur un mode beaucoup plus ouvert, réaliste et bienveillant. Cette réponse harmonieuse peut prendre n’importe quelle forme. Vous êtes libres de répondre de façon appropriée et intelligente, adaptée aux circonstances. Dans tous les cas, la réponse harmonieuse se fera dans l’empathie et la bienveillance plutôt que dans l’agression du conflit émotionnel.\n\n",
                "metadata": {
                    "sort_chapter": 3,
                    "sort_step_nb": 5,
                    "sort_section_nb": 20,
                    "sort_paragraph_nb": 1,
                    "page_title": "PRATIQUE 32",
                    "contents_to_embed_length": 1178,
                    "url": "https://www.openmindfulness.net/20-etape-5-p32/",
                    "source": "3.5.20.01",
                },
            },
            {
                "id": "3.5.20.02",
                "document": " Nous allons développer et expérimenter ces trois étapes :\n\nPremière étape : la reconnaissance et l’accueil\nDescriptif\nLe propos de cet exercice est de reconnaitre la présence de l’émotion. Habituellement, nous ne repérons pas l’arrivée d’une émotion. Si nous pouvions voir l’émotion dès qu’elle apparaît, nous pourrions ne pas nous laisser emporter par son énergie. Pour ce faire, « le tableau de bord corporel » présenté dans la première étape est très utile. Ce que nous avons alors nommé « le baromètre corporel » nous permet de déceler l’émergence d’une émotion à partir de nos sensations corporelles, en étant attentifs à la sensation de notre corps.\nExercice (Durée : 5 mn)\n\nComme d’habitude : installez-vous dans une posture confortable, les sens tout ouverts, détendus dans le regard panoramique, ça respire.\n……\n\n\nDans cet état : sentez votre corps, observez-le. Observez ce qui s’y passe et déceler y les signes d’émergence d’une émotion.\n……\n\n\nNotez particulièrement comment est votre respiration, comment bat votre cœur, les petites sensations dans votre ventre, gorge ou ailleurs, traduisant la présence d’une émotion. Cette attention à la sensation du corps vous aide à reconnaitre l’apparition d’une émotion.\n……\n\n\nSi vous n’avez dans l’instant aucune émotion décelable vous pouvez évoquer une situation émotionnelle de votre vie récente. Vous trouverez certainement facilement une situation qui vous a provoqué une émotion. Evoquez-la et sentez l’émotion.\n……\n\n\nA présent ayant reconnu la présence d’une émotion, vous l’accueillez. Plutôt que de vouloir la suivre ou la fuir, vous accueillez et acceptez simplement sa présence, sans jugement. Vous sentez pleinement son énergie, sa chaleur : cela peut se manifester de différentes façons suivant la nature de l’émotion. Se peut être « la moutarde qui monte au nez », une production accrue d’adrénaline, une respiration plus intense, un rythme cardiaque accéléré… Quoi qu’il en soit, vous accueillez simplement cette émotion et son énergie avec bienveillance, dans une sorte de « oui » qui est comme un sourire bienveillant, un accueil sans jugement.\n……\n\n\nNe vous demandez pas s’il est bien que cette émotion soit là ou pas. Restez simplement dans un état d’attention relâché, accueillant l’émotion, la laissant venir.\n……\nSi cela vous aide, vous pouvez comme pour les pensées avoir recours à la reconnaissance verbale avec l’étiquetage : « émotion » ou encore à la simple reconnaissance non verbale. Cette reconnaissance permet un accueil neutre et bienveillant des émotions, permet d’entrer en amitié avec ce que l’on porte d’émotionnel et de conflictuel en soi.\n……\n\nDeuxième étape : Respirer dans l’émotion, l’incorporant et la laissant se décharger\nDescriptif\nLa deuxième étape consiste à respirer dans l’émotion en l’incorporant et la laissant se décharger dans l’ouvert. Ayant accueilli et accepté la présence de l’émotion nous utilisons l’alternance du cycle respiratoire pour incorporer et laisser se décharger l’émotion. Particulièrement dans l’inspiration nous accueillons son énergie et l’incorporons, faisons corps avec elle. Dans l’expiration nous nous relâchons complètement et laissons cette énergie se décharger dans l’ouvert.\nExercice (Durée : 5 mn)\n\nToujours installé dans la posture confortable, les sens tout ouverts et détendu dans le regard panoramique, ça respire naturellement.\n……\n\n\nA présent, après avoir accueilli l’émotion, vous vous familiarisez avec sa présence et entrez en son énergie sans peur ni résistance. Vous restez dans son ressenti en respirant dans sa sensation.\n……\n\n\nVous incorporez l’émotion, vous vous laissez aller dans la sensation de son énergie, faisant de plus en plus corps avec elle.\n……\n\n\nCette incorporation de l’émotion se vit associée à la respiration, à la pulsation du souffle. Vous vous laissez aller dans celle-ci au rythme de l’inspiration et de l’expiration. Particulièrement, vous associez l’accueil avec l’inspiration et le lâcher prise dans l’ouvert avec l’expiration.\n……\n\n\nEn respirant ainsi, vous vous détendez et vous vous abandonnez dans l’énergie de l’émotion. Dans le va-et-vient de la respiration, progressivement, vous incorporez l’émotion et la laissez se décharger.\n……\n\n\nVous continuez ainsi laissant l’énergie de l’émotion être telle qu’elle est. Dans le lâcher prise associé à la détente la distance entre vous et l’émotion se réduit et finalement disparaît, laissant simplement son énergie dans l’ouvert. L’émotion devient alors une énergie qui n’est pas possédée, une énergie libre qui rayonne et se décharge.\n……\n\n\nEn pratiquant ainsi le temps nécessaire, l’intensité émotionnelle diminue, retombe et finalement se dissout. Le reliquat de son énergie n’est plus conflictuel et peut même devenir source d’une intelligence qui va animer la troisième étape.\n……\n\nTroisième étape : Rester apaisé dans l’état d’observateur abstrait en harmonie avec la situation\nDescriptif\nDans cette troisième étape, une fois que l’émotion s’est déchargée nous  restons tel quel dans la simple présence ouverte globale, en état d’observateur abstrait. Nous sommes alors en harmonie avec la situation et pouvons, si la situation le demande, y répondre de façon adaptée.\n Exercice (Durée : 5 à 10 mn)\n\nToujours installé confortablement, les sens tout ouverts et détendu dans le regard panoramique, ça respire.\n……\n\n\nPour cette troisième étape d’intégration de l’émotion, vous restez apaisé dans l’ouvert, en l’état d’observateur abstrait. Lors de la deuxième étape la charge émotionnelle s’est déchargée, le corps et l’esprit sont maintenant relâchés. Vous restez au repos, tel quel, dans la simple présence ouverte de l’observateur abstrait.\n……\n\n\nNaturellement et tranquillement, ça respire. Vous êtes un, apaisé dans la situation.\n……\n\n\nDans cet état de pleine présence, attentive, ouverte et empathique, vous êtes en harmonie avec la situation et son intelligence, et ainsi, une réponse adaptée peut émerger, que ce soit de dire ou faire quelque chose ou de ne rien dire ni rien faire.\n……\n\n\nAu départ, quand l’émotion était virulente, vous auriez réagi émotionnellement. Mais à présent, l’émotion s’étant déchargée, vous pouvez répondre de façon harmonieuse et empathique, en percevant la situation sur un mode beaucoup plus ouvert, réaliste et bienveillant. Cette réponse harmonieuse peut prendre n’importe quelle forme. Vous êtes libres de répondre de façon appropriée et intelligente, adaptée aux circonstances. Dans tous les cas, la réponse harmonieuse se fera dans l’empathie et la bienveillance plutôt que dans l’agression du conflit émotionnel.\n\n",
                "metadata": {
                    "sort_chapter": 3,
                    "sort_step_nb": 5,
                    "sort_section_nb": 20,
                    "sort_paragraph_nb": 2,
                    "page_title": "PRATIQUE 32",
                    "contents_to_embed_length": 1000,
                    "url": "https://www.openmindfulness.net/20-etape-5-p32/",
                    "source": "3.5.20.02",
                },
            },
            {
                "id": "3.5.20.03",
                "document": "Première étape : la reconnaissance et l’accueil\n\nDescriptif\nLe propos de cet exercice est de reconnaitre la présence de l’émotion. Habituellement, nous ne repérons pas l’arrivée d’une émotion. Si nous pouvions voir l’émotion dès qu’elle apparaît, nous pourrions ne pas nous laisser emporter par son énergie. Pour ce faire, « le tableau de bord corporel » présenté dans la première étape est très utile. Ce que nous avons alors nommé « le baromètre corporel » nous permet de déceler l’émergence d’une émotion à partir de nos sensations corporelles, en étant attentifs à la sensation de notre corps.\nExercice (Durée : 5 mn)\n\nComme d’habitude : installez-vous dans une posture confortable, les sens tout ouverts, détendus dans le regard panoramique, ça respire.\n……\n\n\nDans cet état : sentez votre corps, observez-le. Observez ce qui s’y passe et déceler y les signes d’émergence d’une émotion.\n……\n\n\nNotez particulièrement comment est votre respiration, comment bat votre cœur, les petites sensations dans votre ventre, gorge ou ailleurs, traduisant la présence d’une émotion. Cette attention à la sensation du corps vous aide à reconnaitre l’apparition d’une émotion.\n……\n\n\nSi vous n’avez dans l’instant aucune émotion décelable vous pouvez évoquer une situation émotionnelle de votre vie récente. Vous trouverez certainement facilement une situation qui vous a provoqué une émotion. Evoquez-la et sentez l’émotion.\n……\n\n\nA présent ayant reconnu la présence d’une émotion, vous l’accueillez. Plutôt que de vouloir la suivre ou la fuir, vous accueillez et acceptez simplement sa présence, sans jugement. Vous sentez pleinement son énergie, sa chaleur : cela peut se manifester de différentes façons suivant la nature de l’émotion. Se peut être « la moutarde qui monte au nez », une production accrue d’adrénaline, une respiration plus intense, un rythme cardiaque accéléré… Quoi qu’il en soit, vous accueillez simplement cette émotion et son énergie avec bienveillance, dans une sorte de « oui » qui est comme un sourire bienveillant, un accueil sans jugement.\n……\n\n\nNe vous demandez pas s’il est bien que cette émotion soit là ou pas. Restez simplement dans un état d’attention relâché, accueillant l’émotion, la laissant venir.\n……\nSi cela vous aide, vous pouvez comme pour les pensées avoir recours à la reconnaissance verbale avec l’étiquetage : « émotion » ou encore à la simple reconnaissance non verbale. Cette reconnaissance permet un accueil neutre et bienveillant des émotions, permet d’entrer en amitié avec ce que l’on porte d’émotionnel et de conflictuel en soi.\n……\n\nDeuxième étape : Respirer dans l’émotion, l’incorporant et la laissant se décharger\nDescriptif\nLa deuxième étape consiste à respirer dans l’émotion en l’incorporant et la laissant se décharger dans l’ouvert. Ayant accueilli et accepté la présence de l’émotion nous utilisons l’alternance du cycle respiratoire pour incorporer et laisser se décharger l’émotion. Particulièrement dans l’inspiration nous accueillons son énergie et l’incorporons, faisons corps avec elle. Dans l’expiration nous nous relâchons complètement et laissons cette énergie se décharger dans l’ouvert.\nExercice (Durée : 5 mn)\n\nToujours installé dans la posture confortable, les sens tout ouverts et détendu dans le regard panoramique, ça respire naturellement.\n……\n\n\nA présent, après avoir accueilli l’émotion, vous vous familiarisez avec sa présence et entrez en son énergie sans peur ni résistance. Vous restez dans son ressenti en respirant dans sa sensation.\n……\n\n\nVous incorporez l’émotion, vous vous laissez aller dans la sensation de son énergie, faisant de plus en plus corps avec elle.\n……\n\n\nCette incorporation de l’émotion se vit associée à la respiration, à la pulsation du souffle. Vous vous laissez aller dans celle-ci au rythme de l’inspiration et de l’expiration. Particulièrement, vous associez l’accueil avec l’inspiration et le lâcher prise dans l’ouvert avec l’expiration.\n……\n\n\nEn respirant ainsi, vous vous détendez et vous vous abandonnez dans l’énergie de l’émotion. Dans le va-et-vient de la respiration, progressivement, vous incorporez l’émotion et la laissez se décharger.\n……\n\n\nVous continuez ainsi laissant l’énergie de l’émotion être telle qu’elle est. Dans le lâcher prise associé à la détente la distance entre vous et l’émotion se réduit et finalement disparaît, laissant simplement son énergie dans l’ouvert. L’émotion devient alors une énergie qui n’est pas possédée, une énergie libre qui rayonne et se décharge.\n……\n\n\nEn pratiquant ainsi le temps nécessaire, l’intensité émotionnelle diminue, retombe et finalement se dissout. Le reliquat de son énergie n’est plus conflictuel et peut même devenir source d’une intelligence qui va animer la troisième étape.\n……\n\nTroisième étape : Rester apaisé dans l’état d’observateur abstrait en harmonie avec la situation\nDescriptif\nDans cette troisième étape, une fois que l’émotion s’est déchargée nous  restons tel quel dans la simple présence ouverte globale, en état d’observateur abstrait. Nous sommes alors en harmonie avec la situation et pouvons, si la situation le demande, y répondre de façon adaptée.\n Exercice (Durée : 5 à 10 mn)\n\nToujours installé confortablement, les sens tout ouverts et détendu dans le regard panoramique, ça respire.\n……\n\n\nPour cette troisième étape d’intégration de l’émotion, vous restez apaisé dans l’ouvert, en l’état d’observateur abstrait. Lors de la deuxième étape la charge émotionnelle s’est déchargée, le corps et l’esprit sont maintenant relâchés. Vous restez au repos, tel quel, dans la simple présence ouverte de l’observateur abstrait.\n……\n\n\nNaturellement et tranquillement, ça respire. Vous êtes un, apaisé dans la situation.\n……\n\n\nDans cet état de pleine présence, attentive, ouverte et empathique, vous êtes en harmonie avec la situation et son intelligence, et ainsi, une réponse adaptée peut émerger, que ce soit de dire ou faire quelque chose ou de ne rien dire ni rien faire.\n……\n\n\nAu départ, quand l’émotion était virulente, vous auriez réagi émotionnellement. Mais à présent, l’émotion s’étant déchargée, vous pouvez répondre de façon harmonieuse et empathique, en percevant la situation sur un mode beaucoup plus ouvert, réaliste et bienveillant. Cette réponse harmonieuse peut prendre n’importe quelle forme. Vous êtes libres de répondre de façon appropriée et intelligente, adaptée aux circonstances. Dans tous les cas, la réponse harmonieuse se fera dans l’empathie et la bienveillance plutôt que dans l’agression du conflit émotionnel.\n\n",
                "metadata": {
                    "sort_chapter": 3,
                    "sort_step_nb": 5,
                    "sort_section_nb": 20,
                    "sort_paragraph_nb": 3,
                    "page_title": "PRATIQUE 32",
                    "contents_to_embed_length": 991,
                    "url": "https://www.openmindfulness.net/20-etape-5-p32/",
                    "source": "3.5.20.03",
                },
            },
            {
                "id": "3.5.24.01",
                "document": "Le conseil pour intégrer les émotions dans la pleine présence\n\nLe conseil pour intégrer les émotions dans la pleine présence est de les reconnaitre puis de les incorporer et de les laisser se libérer d’elles-mêmes.\n\n\nLe résumé de l’intégration des émotions dans la pleine présence\n\nLa pratique de la pleine présence dans les émotions nous apprend à les intégrer dans une méthode à trois temps :\n1- Reconnaître les émotions\n2- Les incorporer et les laisser se décharger en respirant avec elles\n3- Répondre harmonieusement en état d’observateur abstrait, en une présence sans saisie en laquelle une réponse adaptée et bienveillante émerge naturellement.\n",
                "metadata": {
                    "sort_chapter": 3,
                    "sort_step_nb": 5,
                    "sort_section_nb": 24,
                    "sort_paragraph_nb": 1,
                    "page_title": "Conseil et résumé pour l'intégration des émotions",
                    "contents_to_embed_length": 103,
                    "url": "https://www.openmindfulness.net/24-etape-5-conseil-et-resume-emotions/",
                    "source": "3.5.24.01",
                },
            },
        ]
        metadata = {
            "cost": {
                "Total Cost (USD)": "$0.05708",
                "Successful Requests": 2,
            },
            "tokens": {
                "Total Tokens": 2854,
                "Prompt Tokens": 2688,
                "Completion Tokens": 166,
            },
        }
    else:
        answer, sources_json, metadata = run_query_with_qa_with_sources(
            query, collection_name=collection_name, response_size=response_size
        )

    sources = format_sources(sources_json, mocked, collection_name)
    # = "Total Tokens: 2854 \nPrompt Tokens: 2688 \nCompletion Tokens: 166 \nSuccessful Requests: 2 \nTotal Cost (USD): $0.05708"
    return answer, sources, metadata, sources_json


def format_sources(sources_json, mocked, collection_name)-> list[str]:
    sources_markdown = ""
    if collection_name == COL_STATE_OF_THE_UNION:
        for source in sources_json:
            sources_markdown += f"""#### {source['id']}
{source['document']}
"""
    elif mocked or collection_name == COL_OPEN_MINDFULNESS:
        # group documents in sources_jon by url from sources["url"], and display for each group the page title from sources["page_title"] and url

        visited_urls = []
        documents = {}
        for source in sources_json:
            metadata = source["metadata"]
            url = metadata["url"]

            if url in documents:
                doc = documents[url]
                doc["contents"] = (
                    doc["contents"] + f"\n\n[...]\n\n {source['document']}"
                )
            else:
                title = f"Chapitre {metadata['sort_chapter']}"
                if metadata["sort_section_nb"] != 0:
                    title += f" Étape {metadata['sort_step_nb']}"
                if metadata["page_title"] != "":
                    title += f": {metadata['page_title']}"
                    
                doc = {"title": title}

                doc["contents"] = f"{source['document']}"

            documents[url] = doc

    return documents


def make_grid(cols, rows):
    grid = [0] * cols
    for i in range(cols):
        with st.container():
            grid[i] = st.columns(rows)
    return grid


# Define main function for Streamlit app
def main():
    st.title("OpenMindfulness Query App")

    language = ""
    token = ""
    st.sidebar.title("Settings")

    # # Add language selector to sidebar
    # language = st.sidebar.selectbox("Language", ["English", "French"])
    is_mocked = st.sidebar.selectbox("is mocked", [True, False])
    if is_mocked:
        collection_name = COL_OPEN_MINDFULNESS
        response_size = ResponseSize.SMALL
    else:
        collection_name = st.sidebar.selectbox(
            "Collection name", [COL_OPEN_MINDFULNESS, COL_STATE_OF_THE_UNION]
        )
        response_size = st.sidebar.selectbox(
            "Response size",
            [ResponseSize.SMALL, ResponseSize.MEDIUM, ResponseSize.LARGE],
        )

    # # Add text input for OpenAPI token to sidebar
    # token = st.sidebar.text_input("OpenAPI Token")

    # Add text input and button for user to enter query
    # query = st.text_input("Enter your query here", "Comment intégrer ses émotions avec la méthode en trois temps ?")
    if collection_name == COL_STATE_OF_THE_UNION:
        query = st.text_input(
            "Enter your query here", "What did the president say about Justice Breyer ?"
        )
    elif collection_name == COL_OPEN_MINDFULNESS:
        query = st.text_input(
            "Enter your query here",
            "Comment intégrer ses émotions avec la méthode en trois temps ?",
        )

    if st.button("Run Query") and query != "":
        answer, sources, metadata, sources_json = run_query(
            query,
            mocked=is_mocked,
            collection_name=collection_name,
            response_size=response_size,
        )
        st.header("Answer")
        st.markdown(answer)
        st.header("Sources")
        tab1, tab2 = st.tabs(["Formatted text", "data"])
        # tab1.markdown(sources)
        for url in sources:
            with tab1.expander(sources[url]["title"]):
                st.markdown(f"[{url}]({url})")
                st.markdown(sources[url]["contents"])
        tab2.json(sources_json)
        st.header("Metadata")
        for k, v in metadata.items():
            # st.subheader(k)
            grid = make_grid(1, len(v.items()))
            for index, (k2, v2) in enumerate(v.items()):
                # if v2 is str and starts with $, then format it as a currency
                if isinstance(v2, str) and '$' in v2:
                    v2 = f"${float(v2.replace('$', '')):.2f}"
                grid[0][index].metric(k2, v2)


# Run main function
if __name__ == "__main__":
    main()
