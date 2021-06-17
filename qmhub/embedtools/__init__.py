import importlib

EMBEDNEAR = {'eed': 'EEd', 'eeq': 'EEq', 'none': 'None'}
EMBEDFAR = {'eeq': 'EEq', 'me': 'ME', 'none': 'None'}
EMBEDMODULE = {'EmbedEEdEEq': '.embed_eed_eeq', 'EmbedEEqEEq': '.embed_eeq_eeq',
               'EmbedEEdME': '.embed_eed_me', 'EmbedEEqME': '.embed_eeq_me',
               'EmbedEEdNone': '.embed_eed_none', 'EmbedEEqNone': '.embed_eeq_none',
               'EmbedNoneNone': '.embed_none_none'}


def choose_embedtool(qmEmbedNear, qmEmbedFar):
    if qmEmbedNear is None:
        qmEmbedNear = 'None'
    try:
        embed_near = EMBEDNEAR[qmEmbedNear.lower()]
    except:
        raise ValueError("Please choose 'eed', 'eeq', or None for qmEmbedNear.")


    if qmEmbedFar is None:
        qmEmbedFar = 'None'
    try:
        embed_far = EMBEDFAR[qmEmbedFar.lower()]
    except:
        raise ValueError("Please choose 'eeq', 'me', or None for qmEmbedFar.")

    embed_class = "".join(['Embed', embed_near, embed_far])

    try:
        embed_module = EMBEDMODULE[embed_class]
    except:
        raise ValueError("Cannot use '{}' for far field while using '{}' for near field.".format(embed_far, embed_near))

    embedtool = importlib.import_module(embed_module, package='qmhub.embedtools').__getattribute__(embed_class)

    return embedtool
