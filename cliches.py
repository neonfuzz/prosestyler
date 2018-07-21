

"""
A list of (lemmatized) cliches and some suggestions for replacements.
"""


# Many thanks to powerthesaurus.org

CLICHES = {
    '11th hour': ['last minute'],
    'a far a the eye can see': [],
    'a good a gold': [],
    'a luck would have prp': [
        'fortuitously', 'fortunately', 'happily',
        'luckily', 'mercifully'],
    'a the crow fly': ['directly', 'immediately', 'straight'],
    'a to whether': ['whether'],
    'about time': [],
    'achilles heel': ['failing', 'frailty', 'vulnerability', 'weakness'],
    'action speak louder than word': [],
    'after prp own heart': [
        'appealing', 'compatible', 'congenial', 'kindred', 'like-minded'],
    'air dirty laundry': ['gossip', 'rumor', 'slander', 'tale'],
    'all in a day\'s work': ['routine', 'typical', 'usual'],
    'all over the map': [
        'chaotic', 'disorganized', 'everywhere', 'ubiquitously'],
    'all talk and no action': [],
    'all that glitter be not gold': [],
    'all that jazz': [],
    'all the help prp can get': [],
    'an eternity': [],
    'anything go': [],
    'arm and a leg': [
        'cost', 'costly', 'excessive', 'exorbitant', 'expense', 'expensive',
        'extravagant', 'overpriced', 'premium'],
    'arm to the teeth': ['armed'],
    'at all time': ['constantly', 'continually', 'invariably', 'perpetually'],
    'at the end of prp rope': ['desperate'],
    'at the end of the day': ['eventually', 'finally', 'ultimately'],
    'at this time': ['currently', 'given that', 'here', 'now'],
    'at wit\'s end': [],
    'baby boomer': ['postwar generation'],
    'back against the wall': [],
    'back seat driver': [],
    'back stabber': [
        'apostate', 'betrayer', 'defector', 'deserter', 'informer', 'rebel',
        'recreant', 'renegade', 'traitor', 'turncoat'],
    'back to the draw board': ['beginning', 'origin', 'outset', 'start'],
    'bad blood': [
        'acrimony', 'animosity', 'animus', 'antagonism', 'antipathy',
        'bitterness', 'conflict', 'enmity', 'hatred', 'hostility',
        'malevolence', 'malice', 'rancor', 'venom'],
    'bad call': [],
    'bad hair day': [],
    'bad seed': [],
    'bait and switch': ['cheat', 'collusion', 'treachery', 'trickery'],
    'bait breath': [
        'agitated', 'anticipating', 'anxious', 'expectant', 'nervous'],
    'ball be in prp court': [],
    'ball roll': [],
    'baptism by fire': [],
    'bare bone': [
        'austere', 'core', 'foundation', 'framework', 'plain', 'spartan',
        'stark', 'unadorned', 'unvarnished'],
    'barge in': [
        'encroach', 'infringe', 'interfere', 'interrupt', 'intrude', 'meddle'],
    'barge right in': [
        'encroach', 'infringe', 'interfere', 'interrupt', 'intrude', 'meddle'],
    'bark up the wrong tree': [],
    'be the case': [],
    'bear down': ['crush', 'drive', 'force', 'press', 'push', 'thrust'],
    'bear yesterday': ['gullible', 'innocent', 'naïve'],
    'beat a dead horse': [
        'assay', 'belabor', 'dwell', 'linger', 'rehash', 'scrutinize'],
    'beat around the bush': ['hedge', 'intimate'],
    'beg to differ': ['disagree', 'dissent', 'oppose'],
    'bell and whistle': [
        'accessories', 'additions', 'embellishments', 'enhancements', 'extras',
        'frills', 'gadgetry', 'luxuries'],
    'bend over backwards': [],
    'bent out of shape': [
        'angry', 'enraged', 'fierce', 'fuming', 'furious', 'indignant',
        'ireful'],
    'best laid plan': [],
    'best thing since slice bread': ['delight', 'pleasure'],
    'bet prp all': ['chance', 'risk', 'venture'],
    'bet prp bottom dollar': [],
    'between a rock and a hard place': [],
    'big a a house': [
        'big', 'colossal', 'enormous', 'gigantic', 'huge', 'immense', 'large',
        'massive', 'monstrous', 'prodigious', 'substantial', 'tremendous'],
    'big wig': ['boss', 'brass', 'executive', 'kingpin'],
    'bigwig': ['boss', 'brass', 'executive', 'kingpin'],
    'bite the hand that feed prp': [
        'abandon', 'betray', 'break faith', 'cross', 'deceive', 'delude',
        'forsake'],
    'both foot on the ground': [
        'businesslike', 'constructive', 'empirical', 'functional', 'operative',
        'practical', 'useful'],
    'bottom line': ['core', 'crux', 'essence', 'gist', 'heart', 'kernel'],
    'bounce back': [
        'backfire', 'echo', 'grow', 'increase', 'rebound', 'recoil', 'recover',
        'recuperate', 'return', 'revive'],
    'brown nose': ['abase', 'beseech', 'court', 'fawn', 'flatter', 'kowtow'],
    'brush off': [
        'discount', 'dismiss', 'disregard', 'ignore', 'rebuff', 'refuse',
        'reject', 'repudiate', 'slight', 'snub', 'spurn'],
    'bun in the oven': ['pregnant'],
    'burn the candle at both end': [
        'lucubrate', 'overdo', 'overextend', 'slave', 'strain', 'strive'],
    'burn the midnight oil': ['elucubrate', 'lucubrate', 'moonlight', 'work'],
    'business a usual': ['as usual', 'as ever'],
    'buy in': [],
    'by the book': ['exact', 'lawfully', 'legally', 'strict', 'strictly'],
    'by the same token': ['futhermore', 'likewise'],
    'by the seat of prp pant': [],
    'ca n\'t stomach': ['hate'],
    'call prp a day': ['cease', 'conclude', 'end', 'finish', 'stop'],
    'call prp a night': ['cease', 'conclude', 'end', 'finish', 'stop'],
    'call the shot': [
        'command', 'control', 'dictate', 'direct', 'govern', 'manage', 'order',
        'oversee', 'regulate', 'rule', 'supervise'],
    'can of worm': [
        'mess', 'problem', 'question', 'situation', 'snafu', 'trouble'],
    'cash cow': ['income', 'source', 'windfall'],
    'cash prp in': [],
    'cat nap': ['nap', 'rest', 'sleep'],
    'cat\'s meow': ['delight', 'pleasure'],
    'cat\'s pajama': ['delight', 'pleasure'],
    'catch on': [
        'apprehend', 'comprehend', 'grasp', 'perceive', 'realize', 'realize'
        'recognize', 'understand'],
    'catnap': ['nap', 'rest', 'sleep'],
    'chew on': [
        'cogitate', 'consider', 'deliberate', 'meditate', 'muse', 'ponder',
        'reflect', 'ruminate', 'think'],
    'child play': ['easy'],
    'chill out': ['calm', 'cool', 'recline', 'relax', 'rest', 'settle'],
    'chip in': ['contribute', 'donate', 'give', 'help', 'participate', 'pay'],
    'clean out': [
        'bankrupt', 'cleanse', 'clear', 'deplete', 'empty', 'evacuate',
        'remove', 'ruin', 'void'],
    'clean slate': ['beginning', 'overthrow', 'overturn', 'start', 'upset'],
    'clear the air': [],
    'close call': [],
    'coast to coast': [],
    'cold blood': [
        'aloof', 'apathetic', 'cold', 'cool', 'impassible', 'impassive',
        'indifferent', 'insensitive', 'lukewarm', 'pococurante', 'senseless'
        'unconcerned', 'uninterested'],
    'cold foot': [
        'alarm', 'anxiety', 'apprehension', 'dismay', 'dread', 'fear',
        'fearfulness', 'fright', 'horror', 'panic', 'terror', 'trepidation'],
    'cold hearted': [
        'cool', 'disaffected', 'dull', 'estranged', 'flat', 'frigid',
        'languid', 'obtuse', 'phlegmatic'],
    'cold shoulder': [
        'dismissal', 'disregard', 'rebuff', 'rebuke', 'rejection', 'slight',
        'snub', 'spurn'],
    'come across': ['discover', 'encounter', 'find', 'meet', 'uncover'],
    'cookie cutter': [
        'alike', 'analogous', 'homogenous', 'identical', 'indistinguishable'
        'interchangeable', 'similar', 'standard', 'twin', 'undifferentiated'
        'uniform'],
    'copasetic': [
        'acceptable', 'congenial', 'decent', 'fair', 'gratifying', 'palatable',
        'satisfactory'],
    'crack of dawn': [
        'dawn', 'daybreak', 'daylight', 'early morning', 'first light',
        'sunrise', 'sunup'],
    'cream of the crop': [
        'best', 'choice', 'elect', 'elite', 'favorite', 'pick', 'prime',
        'prize', 'top'],
    'cross the line': [],
    'cut edge': [
        'change', 'forefront', 'innovation', 'innovative', 'latest', 'new',
        'newest', 'novelty', 'vanguard'],
    'dead ringer': [
        'clone', 'copy', 'double', 'duplicate', 'facsimile', 'likeness',
        'match', 'replica'],
    'deal with': [
        'answer', 'confront', 'control', 'cope', 'dispose of', 'face',
        'handle', 'manage', 'manipulate', 'tackle', 'treat', 'troubleshoot'],
    'death and destruction': [],
    'dirt cheap': [
        'bargain', 'cheap', 'economical', 'inexpensive', 'reasonable'],
    'do n\'t hold prp breath': ['never', 'improbable'],
    'do n\'t look back': [],
    'do or die': [
        'formal', 'inelastic', 'inexorable', 'intractable', 'reserved',
        'resolute', 'tough'],
    'do time': ['imprisoned'],
    'do what prp take': [],
    'doe n\'t stand a chance': [],
    'down and out': [
        'beaten', 'broke', 'defeated', 'destitute', 'impoverished', 'needy',
        'penniless', 'poor', 'ruined', 'vagabond'],
    'down on prp luck': [
        'afflicted', 'cursed', 'luckless', 'penniless', 'unfortunate',
        'unlucky'],
    'draw a blank': ['forget'],
    'dream on': [],
    'dress to kill': ['dapper', 'fashionable', 'formal', 'well-dressed'],
    'dress to the nine': ['dapper', 'fashionable', 'formal', 'well-dressed'],
    'drop of a hat': [
        'easily', 'freely', 'immediately', 'instantly', 'promptly', 'quickly',
        'rapidly', 'readily', 'speedily', 'suddenly'],
    'due to the fact that': ['because'],
    'elephant in the room': [],
    'eleventh hour': ['last minute'],
    'emotional roller coaster': [],
    'end justifes the mean': [],
    'end justify the mean': [],
    'every detail': ['everything'],
    'every little detail': ['everything'],
    'every little thing': ['everything'],
    'every single': ['each'],
    'fair weather friend': ['phony', 'hypocrite'],
    'fall through the crack': [],
    'fan the flame': [
        'aggravate', 'agitate', 'arouse', 'burn', 'excite', 'feed', 'incense',
        'nettle', 'pique', 'provoke', 'stir'],
    'fancy meeting prp here': [],
    'far cry': [
        'conflict', 'departure', 'disagreement', 'discordance', 'discrepany'
        'disparate', 'dissent', 'dissimilar', 'dissonance', 'distinctive',
        'divergence', 'opposition', 'variation'],
    'fatal blow': [],
    'figure prp out': [
        'appreciate', 'comprehend', 'follow', 'learn', 'realize', 'recognize',
        'understand', 'visualize'],
    'filthy rich': [
        'affluent', 'flush', 'moneyed', 'opulent', 'prosperous', 'rich',
        'wealthy'],
    'fine line': [],
    'finger cross': ['hopefully'],
    'fire on all cylinder': [],
    'first thing first': [],
    'first thing\'s first': [],
    'fleet foot': ['fast', 'urgent', 'swift', 'flying', 'breakneck'],
    'flip flop': [],
    'fly on the wall': ['observer', 'onlooker', 'spectator', 'witness'],
    'follow in prp footstep': [
        'assume', 'borrow', 'duplicate', 'emulate', 'follow', 'model',
        'parallel'],
    'follow the leader': [],
    'fool gold': ['foolishness'],
    'for all intent and purpose': [
        'apparently', 'approximately', 'basically', 'essentially', 'nearly',
        'ostensibly', 'practically'],
    'for cry out loud': [],
    'for the purpose of': ['to'],
    'free a a bird': [
        'free', 'relaxed', 'unbound', 'unconfined', 'unencumbered'],
    'free reign': [],
    'frighten to death': [
        'aghast', 'awestruck', 'breathless', 'frightened', 'pale', 'scared'],
    'from day one': ['beginning', 'firstly', 'initially', 'originally'],
    'full of hot air': ['chattering', 'garrulous', 'gossping', 'long-winded'],
    'funny business': [
        'deceit', 'deception', 'duplicity', 'fraud', 'guile', 'horseplay',
        'nonsense', 'trickery'],
    'get a life': [],
    'get a room': [],
    'get down': [
        'alight', 'deject', 'demoralize', 'depress', 'descend', 'dismount',
        'dispirit', 'humble', 'lower', 'slip'],
    'get enough of': [],
    'get lose': ['depart', 'exit', 'leave', 'move'],
    'get off on the wrong foot': [],
    'get off': ['get away', 'exit', 'leave', 'depart', 'quit', 'retire'],
    'get prp': [
        'appreciate', 'comprehend', 'follow', 'learn', 'realize', 'recognize',
        'understand', 'visualize'],
    'get to the bottom of': [
        'comprehend', 'decipher', 'decode', 'discover', 'disentangle',
        'fathom', 'infer'],
    'give and take': [
        'alternate', 'bargain', 'barter', 'commute', 'compromise', 'exchange',
        'interchange', 'reciprocate', 'share', 'swap', 'switch', 'trade'],
    'give prp a rest': ['quiet', 'silent'],
    'give prp a whirl': ['try'],
    'glimmer of hope': [],
    'gloss over': [
        'conceal', 'cover up', 'disregard', 'dodge', 'hide', 'ignore',
        'neglect', 'overlook'],
    'glutton for punishment': ['masochist'],
    'go against the grain': [
        'bother', 'disagree', 'distress', 'hurt', 'injure', 'trouble',
        'upset'],
    'go crazy': [],
    'go figure': [],
    'go over prp head': [],
    'go overboard': ['exaggerate', 'overdo', 'overindulge'],
    'go too far': ['overdo', 'overstep'],
    'go with prp gut': [],
    'golden child': [],
    'good call': [],
    'grasp at straw': [],
    'grass be always greener': [],
    'green with envy': [
        'bitter', 'covetous', 'envious', 'grudging', 'invidious', 'jealous',
        'resentful', 'suspicious'],
    'grin and bear prp': ['bear', 'endure', 'perservere'],
    'ground rule': ['axiom', 'guideline', 'stipulation'],
    'grow like a weed': [],
    'gut check': [],
    'ha the ability to': ['can'],
    'hand on': [
        'active', 'applied', 'experienced', 'experiential', 'experimental',
        'manual', 'practical'],
    'hang in there': ['bear', 'endure', 'persevere'],
    'hang on every word': ['be attentive'],
    'happily ever after': [],
    'happy a a': [
        'blissful', 'cheerful', 'delighted', 'ecstatic', 'glad', 'joyful',
        'joyous', 'merry', 'pleased'],
    'happy camper': [
        'blissful', 'cheerful', 'delighted', 'ecstatic', 'glad', 'joyful',
        'joyous', 'merry', 'pleased'],
    'hard head': [
        'determined', 'determined', 'obstinate', 'obstinate', 'resolute',
        'shrewd', 'stubborn', 'unyielding'],
    'hard pill to swallow': [],
    'hard to swallow': ['far-fetched', 'forced', 'improbable'],
    'hardheaded': [
        'determined', 'determined', 'obstinate', 'obstinate', 'resolute',
        'shrewd', 'stubborn', 'unyielding'],
    'hare brain': [
        'fickle', 'foolish', 'frivolous', 'giddy', 'silly', 'stupid',
        'trivial', 'unrealistic'],
    'harebrained': [
        'fickle', 'foolish', 'frivolous', 'giddy', 'silly', 'stupid',
        'trivial', 'unrealistic'],
    'head over heel': [
        'completely', 'deeply', 'headlong', 'passionately', 'utterly',
        'wildly'],
    'head up': ['alert', 'bright', 'careful', 'lively', 'sharp', 'wary'],
    'heart breaker': [],
    'heart breaking': [],
    'heart of gold': ['benign', 'compassionate', 'gracious', 'kind', 'tender'],
    'high hope': ['anticipation', 'aspiration', 'promise', 'prospect'],
    'history repeat prp': [],
    'hit below the belt': [
        'beguile', 'burn', 'cheat', 'defraud', 'dupe', 'fleece', 'mislead',
        'trick'],
    'hit on': ['catch', 'discover', 'encounter', 'find', 'flirt', 'spot'],
    'hit the book': ['cram', 'learn', 'read', 'study'],
    'hit the road': ['begin', 'depart', 'leave', 'set out', 'start', 'travel'],
    'hold a candle to': ['approach', 'equal', 'match', 'parallel'],
    'ill fat': [
        'doomed', 'foreboding', 'hapless', 'inauspicious', 'menacing',
        'sinister', 'threatening', 'unfortunate', 'unlucky'],
    'important to note': ['important'],
    'in due time': ['finally', 'inevitably', 'soon', 'ultimately'],
    'in order to': ['to'],
    'in prp opinion': [],
    'in spite of the fact': ['despite', 'although'],
    'in spite of': ['despite', 'although'],
    'in term of': ['about', 'regarding'],
    'in the clear': [],
    'in the event of': ['if'],
    'in the event': ['if'],
    'in the process of': ['while', 'when'],
    'jack of all trade': ['multitalented', 'versatile'],
    'judge a book by prp cover': [],
    'jury be out': [],
    'jury be still out': [],
    'just a minute': [],
    'just a second': [],
    'just dessert': [
        'comeuppance', 'justice', 'penalty', 'punishment', 'recompense',
        'redress', 'reprisal', 'requital', 'retribution', 'vengeance'],
    'keep prp down': [
        'dampen', 'muffle', 'mute', 'quash', 'stifle', 'subdue', 'supress'],
    'king\'s ransom': ['sum', 'treasure'],
    'knock off': [
        'copy', 'discount', 'dupe', 'duplicate', 'facsimile', 'fake',
        'imitation', 'likeness', 'markdown', 'model', 'reduction', 'replica',
        'reproduction'],
    'knock on wood': ['believe', 'long for'],
    'know where prp stand': [],
    'last but not least': ['finally', 'in conclusion'],
    'last ditch effort': [],
    'last minute': [],
    'less of two evil': [],
    'let bygone be bygone': ['absolve', 'condone', 'excuse', 'forget'],
    'like a glove': [],
    'line in the sand': [],
    'litmus test': ['gauge', 'measure', 'test'],
    'live and learn': [],
    'long shot': ['fluke', 'prospect', 'underdog'],
    'long time no see': ['good morrow', 'how are you', 'nice to see you'],
    'look a gift horse in the mouth': [],
    'look before prp leap': [],
    'look on the bright side': ['hope'],
    'lose track of time': [],
    'make end meet': [
        'conserve', 'cope', 'economize', 'manage', 'restrain', 'save',
        'scrimp', 'skimp', 'survive'],
    'make head or tail of': [
        'comprehend', 'decipher', 'decode', 'discover', 'disentangle',
        'fathom', 'infer'],
    'make or break': [],
    'make sense': ['correspond', 'be consistent', 'cohere'],
    'make up': [
        'complete', 'compose', 'comprise', 'concoct', 'constitute',
        'construct', 'contrive', 'create', 'devise', 'fabricate', 'form',
        'formulate', 'invent', 'make'],
    'many hand make light work': [],
    'matter of fact': [
        'correct', 'emotionless', 'lifeless', 'obdurage', 'plain',
        'practicality', 'proper', 'prosaic', 'sober', 'unimaginative'],
    'matter of time': ['certainty', 'eventuality', 'inevitability'],
    'mission critical': [],
    'more likely to': ['appropriate', 'inclined', 'plausible', 'probable'],
    'more than anything': ['acutely', 'avidly', 'intensely', 'vehemently'],
    'more than life prp': [],
    'more than meet the eye': [],
    'more the merrier': [],
    'murphy\'s law': [],
    'near and dear': [],
    'necessary evil': ['essential', 'requisite', 'unavoidable'],
    'needle in a haystack': [
        'fruitless', 'ineffective', 'ineffectual', 'pointless', 'ridiculous'
        'ridiculous', 'senseless', 'silly', 'unavailing'],
    'needle to say': ['clearly', 'consequently', 'naturally', 'obviously'],
    'nick of time': [],
    'night and day': [
        'constantly', 'continually', 'continuously', 'incompatible',
        'opposites', 'perennially', 'perpetually'],
    'nip prp in the bud': [
        'anticipate', 'arrest', 'control', 'cut short', 'foil', 'interrupt',
        'preclude', 'prevent', 'stop', 'stymie'],
    'no accounting for taste': [],
    'no brainer': ['easy', 'obvious'],
    'no doubt': [
        'certainly', 'indubitably', 'likely', 'most', 'probably',
        'undoubtedly', 'unquestionably'],
    'no go': [
        'abstract', 'failure', 'futile', 'impossible', 'impotent',
        'impractical', 'ineffective', 'ineffectual', 'superfluous', 'useless',
        'useless'],
    'no hold bar': [
        'absolutely', 'boundless', 'candidly', 'definitely', 'downright',
        'extremely', 'frankly', 'honestly', 'limitless', 'outright',
        'unequivocally', 'uninhibited'],
    'no string attach': [],
    'no time like the present': [],
    'nothing personal': [],
    'nothing to sneeze at': [],
    'nothing to write home about': [
        'average', 'boring', 'commonplace', 'dull', 'mediocre', 'normal',
        'pedestrian', 'prosaic', 'typical', 'undistinguished', 'uneventful',
        'unexceptional', 'unmemorable'],
    'now or never': [],
    'off guard': ['oblivious', 'unawares', 'unready', 'unsuspecting'],
    'off kilter': [
        'amiss', 'askew', 'bizarre', 'cranky', 'crooked', 'curious',
        'eccentric', 'eratic', 'funny', 'listing', 'lopsided', 'offbeat',
        'peculiar', 'quirky', 'skewed', 'slanted', 'slantwise', 'strange',
        'tilted', 'uneven', 'weird'],
    'off the cuff': ['impromptu', 'improvised', 'spontaneous'],
    'off the hook': ['free', 'exempt'],
    'off the top of prp head': [
        'carelessly', 'immediate', 'impromptu', 'spontaneous', 'unplanned',
        'unrehearsed'],
    'old school': [
        'anachronistic', 'ancient', 'antiquated', 'antique', 'backward',
        'conformist', 'conservative', 'conventional', 'customary', 'old',
        'outmoded', 'past', 'purist', 'quaint', 'traditional', 'vintage'],
    'on a limb': [
        'alarming', 'dangerous', 'erratic', 'hazardous', 'insecure',
        'perilous', 'risky', 'unsafe'],
    'on a roll': [
        'auspicious', 'blessed', 'favored', 'fortuitous', 'fortunate', 'hot'
        'lucky', 'promising', 'winning'],
    'on prp toe': [
        'alert', 'attentive', 'aware', 'bright', 'careful', 'lively',
        'observant', 'perceptive', 'quick', 'ready', 'sharp', 'vigilant',
        'watchful'],
    'on the back burner': [
        'deferred', 'delayed', 'interrupted', 'postpone', 'suspended'],
    'on the ball': [
        'able', 'adept', 'alert', 'astute', 'attentive', 'aware', 'clever',
        'competent', 'keen', 'perceptive', 'ready', 'sharp', 'shrewd', 'smart',
        'vigilant', 'watchful'],
    'on the fly': [],
    'on the same page': ['in agreement', 'in accord'],
    'on track': ['correct', 'right', 'successful'],
    'once again': ['again'],
    'one fell swoop': [
        'as one', 'collectively', 'combined', 'concertedly', 'concurrently',
        'conjointly', 'en masse', 'jointly', 'mutually', 'together'],
    'one in million': [
        'blessing', 'marvel', 'prodigy', 'rarity', 'sensation', 'spectacle',
        'winner', 'wonder', 'wunderkind'],
    'one night stand': [
        'affair', 'date', 'engagement', 'meeting', 'rendezvous', 'tryst'],
    'open book': [],
    'out of pocket': [],
    'out of the wood': [
        'better', 'cured', 'healed', 'healthier', 'mending', 'recovered',
        'well'],
    'out of whack': ['defective', 'disrepair', 'down', 'ruined'],
    'out to lunch': [],
    'over and over': ['frequently', 'often'],
    'over the top': [
        'ardent', 'excessive', 'fanatical', 'fervent', 'feverish', 'manic',
        'obsessive', 'overzealous', 'too much'],
    'pale in comparison': [],
    'pandora\'s box': [],
    'par for the course': [],
    'pass away': [
        'decease', 'deliver', 'die', 'expire', 'give', 'impart', 'perish',
        'succumb', 'transfer', 'transmit'],
    'pass on': [
        'decease', 'deliver', 'die', 'expire', 'give', 'impart', 'perish',
        'succumb', 'transfer', 'transmit'],
    'path of least resistance': [],
    'peachy': [
        'divine', 'excellent', 'fine', 'great', 'keen', 'lovely', 'marvelous',
        'neat', 'pleasant', 'quality', 'satisfactory', 'superb'],
    'piece of cake': ['easy', 'simple'],
    'plain and simple': ['austere', 'plain', 'simple'],
    'play prp by ear': ['improvise'],
    'play the field': ['dally', 'date', 'flirt', 'gallivant', 'philander'],
    'play with fire': ['chance', 'endanger', 'hazard', 'risk', 'venture'],
    'press for time': ['breathless', 'hurried'],
    'prp bad': ['excuse me', 'I\'m sorry'],
    'prp be possible': ['can', 'could', 'may', 'might'],
    'prp can imagine': [],
    'prp hand be tie': ['helpless', 'indefensible', 'powerless'],
    'push the envelope': ['forge ahead', 'pioneer'],
    'put 2 and 2 together': [],
    'put two and two together': [],
    'quiet before the storm': [],
    'rain on prp parade': [],
    'raise the bar': [],
    'read between the line': ['deduce', 'infer', 'interpret'],
    'read the fine print': [],
    'red carpet': ['greeting', 'reception', 'welcome'],
    'red herring': ['con', 'diversion', 'feint', 'ploy', 'ruse'],
    'rhyme or reason': ['rationale'],
    'right up prp alley': [],
    'ring a bell': ['calls to mind', 'elicit', 'evoke', 'recall'],
    'road less travel': [],
    'rock the boat': ['disturb', 'trouble', 'upset'],
    'rocket science': [],
    'rough around the edge': [],
    'rub prp the wrong way': [
        'aggravate', 'anger', 'annoy', 'bother', 'disturb', 'enrage', 'grate',
        'infuriate', 'irk', 'irritate', 'needle', 'nettle', 'rattle', 'try'],
    'rule of thumb': [
        'guideline', 'guidepost', 'standard', 'principle', 'rule'],
    'run circle around': [
        'beat', 'circumvent', 'defeat', 'devastate', 'exceed', 'floor',
        'outdo', 'overrun', 'overwhelm', 'shatter', 'thrash', 'top', 'wreck'],
    'run for prp money': [],
    'rush for time': ['breathless', 'hurried'],
    'same exact': ['same'],
    'say what prp will': ['interject'],
    'scar prp to death': [
        'afraid', 'aghast', 'ashen', 'blanched', 'cowed', 'frozen', 'horrifed',
        'intimidated', 'pallid', 'paralysed', 'petrified', 'scared', 'stunned',
        'terrified', 'unnerved'],
    'scar to death': [
        'afraid', 'aghast', 'ashen', 'blanched', 'cowed', 'frozen', 'horrifed',
        'intimidated', 'pallid', 'paralysed', 'petrified', 'scared', 'stunned',
        'terrified', 'unnerved'],
    'second wind': [],
    'see eye to eye': ['acquiesce', 'agree with', 'comply', 'concur'],
    'see the light': ['learn', 'reform'],
    'sell out': ['apostatize', 'betray', 'cross', 'defect', 'desert'],
    'sensory overload': [],
    'shed light on': ['clarify', 'elucidate', 'illuminate', 'illustrate'],
    'shed some light on': ['clarify', 'elucidate', 'illuminate', 'illustrate'],
    'short fuse': ['fury', 'ire', 'sensitivity', 'temper', 'temperament'],
    'silver lining': ['bright side'],
    'sit tight': ['bide', 'delay', 'linger', 'loiter', 'stay', 'wait'],
    'slippery slope': [],
    'slow and steady win the race': [],
    'smoke and mirror': ['deceit'],
    'speed of light': ['quick', 'rapid', 'speedy', 'swift'],
    'split second': ['instant'],
    'spread the news': ['advertise', 'broadcast', 'circulate', 'disseminate'],
    'spread the word': ['advertise', 'broadcast', 'circulate', 'disseminate'],
    'spruce up': ['clean', 'neaten', 'straighten', 'tidy'],
    'square one': ['beginning', 'origin', 'outset', 'start'],
    'start off': ['begin', 'commence', 'launch', 'start'],
    'state of the art': [
        'advanced', 'contemporary', 'fashionable', 'latest', 'modern', 'new'
        'novel', 'popular'],
    'sticky subject': [],
    'stress out': [],
    'sugarcoat': [
        'gloss', 'minimize', 'mollify', 'qualify', 'soften', 'temper'],
    'survival of the fit': ['natural selection'],
    'sweeten the pot': [],
    'take a breather': ['rest'],
    'take one for the team': [],
    'take prp from prp': [],
    'take stock': ['check', 'inspect', 'inventory', 'scrutinize'],
    'take the cake': ['dominate', 'triumph', 'win'],
    'take the easy way out': [],
    'take the plunge': [
        'dare', 'decide', 'initiate', 'launch', 'start', 'try', 'undertake'],
    'talk shop': [],
    'team player': ['collaborator'],
    'tempt fate': ['endanger', 'gamble', 'imperil', 'jeopardize', 'risk'],
    'test the water': [
        'check', 'inquire', 'inspect', 'investigate', 'probe', 'query',
        'research', 'scrutinize', 'study', 'survey', 'test', 'verify'],
    'that be say': [],
    'the eye of the beholder': [],
    'the fact': [],
    'the whole ball of wax': [
        'aggregate', 'all', 'entirety', 'everything', 'gross', 'sum', 'total',
        'totality', 'whole', 'work'],
    'there have be': [],
    'thick head': [],
    'think outside the box': [],
    'third wheel': ['superfluous', 'surplus'],
    'tickle pink': [
        'delighted', 'elated', 'euphoric', 'glad', 'happy', 'joyful',
        'overjoyed', 'pleased', 'thrilled'],
    'tie the knot': ['marry', 'wed'],
    'time after time': ['frequently', 'often'],
    'time and again': ['frequently', 'often'],
    'time be money': [],
    'time of day': ['time'],
    'time of prp life': ['transformative'],
    'time will tell': [],
    'tip of the iceberg': ['exterior', 'perfunctory', 'superficial'],
    'to each prp own': [],
    'to this day': ['thus far'],
    'tongue in cheek': ['insincere', 'ironically', 'mocking'],
    'too much information': [],
    'tooth and nail': ['hard', 'strong'],
    'top dog': ['chief', 'director', 'leader'],
    'touch and go': [],
    'tread lightly': [],
    'trial by fire': [],
    'tried and true': [
        'dependable', 'proved', 'reliable', 'safe', 'sure', 'trustworthy'],
    'tune out': ['disregard', 'ignore', 'neglect', 'overlook'],
    'two face': [
        'artificial', 'backhanded', 'calculating', 'conniving', 'crooked',
        'deceitful', 'deceptive', 'disingenuous', 'duplicitous', 'false',
        'fickle', 'fraudulent', 'hypocritical', 'insincere', 'perfidious',
        'scheming', 'treacherous', 'untrustworthy'],
    'ultimate price': [],
    'up for grab': ['at hand', 'free', 'pending'],
    'up the ante': [],
    'uphill battle': ['formidable', 'herculean', 'involved'],
    'wake up call': [],
    'wash up': ['defeated', 'ruined'],
    'waste of time': ['bore', 'bother'],
    'wear many hat': ['resourceful', 'versatile'],
    'wear prp heart out on prp sleeve': [],
    'well half': [
        'bedmate', 'boyfriend', 'consort', 'girlfriend', 'husband',
        'inamorata', 'inamorato', 'mate', 'partner', 'significant other',
        'spouse', 'sweetheart', 'wife'],
    'well late than never': [],
    'well off': ['affluent', 'prosperous', 'rich', 'wealthy'],
    'well safe than sorry': [],
    'well than ever': [],
    'whatever float prp boat': [],
    'when all be say and do': [],
    'wing prp': ['improvise'],
    'winning combination': [],
    'wipe the slate clean': ['absolve', 'acquit'],
    'witch hunt': ['stigmatization'],
    'with a grain of salt': ['doubtfully', 'skeptically'],
    'with fly color': ['well'],
    'with reference to': ['about', 'concerning', 'regarding'],
    'with regard to': ['about', 'concerning', 'regarding'],
    'work up': [
        'agitated', 'aroused', 'distraught', 'emotional', 'excited',
        'feverish', 'frantic', 'frenetic', 'frenzied', 'hysterical',
        'impassioned', 'irate', 'jittery', 'nervous', 'overwrought', 'tense',
        'thrilled', 'upset'],
    }
