
def calculate_PDI(ego_wc, total_wc):
    """
    Calculate the participant dominance index by taking the ratio between the words contributed
    by ego (donor) and an alter (contact) This metric results in values
    betweeen 0 and 1. Zero indicates exclusive alter, while 1 indicates exclusive ego contribution. 
    """
    import numpy as np

    values = np.divide(ego_wc,total_wc)
    validate_values(values,0,1,variable='PDI')
    return values
    
def validate_values(x, range_min, range_max,variable=''):
    """Validate that all values are within the expected range."""
    import numpy as np
    if not isinstance(x, (list, np.ndarray)):
        x = [x]
    if not all((range_min <= item <= range_max) or np.isnan(item) for item in x):
        raise ValueError("The {variable} values are not within the expected range or are nan.")

def calculate_rGini(intensity):
    """
    Calculate the reversed Gini index (1 - Gini coefficient) for an array.
    The Gini index is adapted from Alshamsi et al. 2016: 
    "Network Diversity and Affect Dynamics: The Role of Personality Traits"
    
    Parameters
    ----------
    intensity : array-like
        The intensity of the donor's interactions (e.g., total words exchanged) in different chats.

    Returns
    -------
    float
        The reversed Gini index (1 - Gini coefficient, 0 = maximum inequality, 1 = maximum equality).
    """
    import numpy as np
    
    k = len(intensity)
    intensity = sorted(intensity)
    product = []
    for ind,item in enumerate(intensity):
        product.append(item*(ind+1))
    index = (2*np.sum(product)/(k*np.sum(intensity)))-((k+1)/k)
    validate_values(index, 0, 1,variable='Gini index')
    return 1-index


def shannon_entropy_equality(intensity):
    """
    Compute the normalized Shannon entropy to measure chat equality.
    
    Parameters
    ----------
    intensity : array-like
        The intensity of the donor's interactions (e.g., total words exchanged) in different chats.

    Returns
    -------
    float
        Normalized entropy score (0 = maximum inequality, 1 = maximum equality).
    """
    import numpy as np

    word_counts = np.array(intensity)
    total_words = np.sum(intensity)
    
    # Avoid division by zero if there are no messages
    if total_words == 0:
        return 0

    # Compute proportions
    p = word_counts / total_words
    
    # Compute Shannon entropy
    entropy = -np.sum(p * np.log2(p + 1e-10))  # Small epsilon to avoid log(0)
    
    # Normalize by maximum entropy (log2(N)) to get a 0-1 scale
    max_entropy = np.log2(len(word_counts))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    return normalized_entropy

def hhi_equality(intensity):
    """
    Compute the Herfindahl-Hirschman Index (HHI) and normalize it to measure chat equality.
    
    Parameters
    intensity : array-like
        The intensity of the donor's interactions (e.g., total words exchanged) in different chats.

    Returns
    -------
    float
        Normalized HHI-based equality score (0 = maximum inequality, 1 = maximum equality).
    """
    import numpy as np
    word_counts = np.array(intensity)
    total_words = np.sum(intensity)
    
    # Avoid division by zero if there are no messages
    if total_words == 0:
        return 0

    # Compute proportions
    p = word_counts / total_words
    
    # Compute HHI (sum of squared proportions)
    hhi = np.sum(p**2)
    
    # Normalize so that higher values indicate more equality
    min_hhi = 1 / len(word_counts)  # Minimum possible HHI (perfect equality)
    max_hhi = 1.0  # Maximum possible HHI (one dominant chat)
    
    normalized_hhi = (max_hhi - hhi) / (max_hhi - min_hhi)  # Scale to 0-1 range

    return normalized_hhi

def normalized_entropy(probabilities):
    from scipy.stats import entropy
    import numpy as np
    
    # Compute Shannon entropy
    shannon_entropy = entropy(probabilities, base=2)

    # Compute maximum possible entropy (log2 of the number of bins)
    max_entropy = np.log2(len(probabilities))

    # Normalize entropy (scale between 0 and 1)
    normalized_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0
    validate_values(normalized_entropy, 0, 1,variable='Normalized entropy')
    return normalized_entropy

def plateau_score(intensity):
    import numpy as np
    intensity = np.array(intensity)
    proportions = intensity/np.sum(intensity)
    assert np.isclose(proportions.sum(), 1.0), "Proportions do not sum to 1"
    
    top_chat = proportions.max()
    return 1-top_chat


def response_times(messages,ego):
    messages = messages.sort_values(by='datetime')
    recent_sender = messages.iloc[0] # start with the first person in the chat
    alter_times = []
    ego_times = []
    for index, row in messages.iterrows():    
        if row['sender_id'] != recent_sender['sender_id']: # make sure the sender has changed
           delta = row['datetime']-recent_sender['datetime']
           delta = delta.total_seconds()
           if row['sender_id'] == ego:
               ego_times.append(delta)
           else:
               alter_times.append(delta)
        recent_sender = row
    return ego_times,alter_times