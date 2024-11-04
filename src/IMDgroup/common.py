"""Common routines used across the sub-libraries.
"""
import logging

logger = logging.getLogger(__name__)


def groupby_cmp(lst, cmp_eq, title_function=None):
    """Group elements of list LST comparing them using CMP_EQ function.
    CMP_EQ should take two elements of the list as arguments and
    return True iff they are equal.
    Optional TITLE_FUNCTION argument is a function used to generate
    item name in the logs.
    Return a list of grouped lists.
    """
    groups = []

    def _add_to_groups(item):
        """Add ITEM to groups.
        If ITEM is not the same with all items in groups,
        create a separate group.
        Modifies "groups" by side effect.
        """
        in_group = False
        for group in groups:
            for group_item in group:
                if cmp_eq(item, group_item):
                    logger.debug(
                        'Appending %s to existing group',
                        title_function(item) if title_function is not None
                        else "??"
                    )
                    group.append(item)
                    in_group = True
                    break
            if in_group:
                break
        if not in_group:
            logger.debug(
                'Creating a new group for %s',
                title_function(item) if title_function is not None
                else "??"
            )
            # pylint: disable=modified-iterating-list
            groups.append([item])

    for item in lst:
        _add_to_groups(item)

    return groups
