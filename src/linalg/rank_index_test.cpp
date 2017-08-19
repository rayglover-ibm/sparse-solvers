#include <linalg/rank_index.h>

#include <gtest/gtest.h>

TEST(rank_index, insert)
{
    ss::rank_index<char> ranks;
    EXPECT_EQ(0, ranks.size());

    EXPECT_EQ(0, ranks.insert('a'));
    EXPECT_EQ(1, ranks.size());
    EXPECT_EQ(0, ranks.rank_of('a'));
    EXPECT_EQ(-1, ranks.rank_of('c'));

    EXPECT_EQ(1, ranks.insert('c'));
    EXPECT_EQ(2, ranks.size());
    EXPECT_EQ(0, ranks.rank_of('a'));
    EXPECT_EQ(1, ranks.rank_of('c'));

    EXPECT_EQ(1, ranks.insert('b'));
    EXPECT_EQ(3, ranks.size());
    EXPECT_EQ(0, ranks.rank_of('a'));
    EXPECT_EQ(1, ranks.rank_of('b'));
    EXPECT_EQ(2, ranks.rank_of('c'));

    EXPECT_EQ(1, ranks.insert('b'));
    EXPECT_EQ(3, ranks.size());
}

TEST(rank_index, erase)
{
    ss::rank_index<char> ranks;
    EXPECT_EQ(0, ranks.size());

    ranks.insert('a');
    ranks.insert('d');
    ranks.insert('b');
    ranks.insert('c');

    EXPECT_EQ('a', ranks.rank_at(0));
    EXPECT_EQ('b', ranks.rank_at(1));
    EXPECT_EQ('c', ranks.rank_at(2));
    EXPECT_EQ('d', ranks.rank_at(3));

    EXPECT_EQ(4, ranks.size());
    EXPECT_FALSE(ranks.erase('z'));
    EXPECT_TRUE(ranks.erase('b'));

    EXPECT_EQ(3, ranks.size());
    EXPECT_EQ(0, ranks.rank_of('a'));
    EXPECT_EQ(-1, ranks.rank_of('b'));
    EXPECT_EQ(1, ranks.rank_of('c'));
    EXPECT_EQ(2, ranks.rank_of('d'));

    EXPECT_TRUE(ranks.erase('a'));

    EXPECT_EQ(2, ranks.size());
    EXPECT_EQ(-1, ranks.rank_of('a'));
    EXPECT_EQ(-1, ranks.rank_of('b'));
    EXPECT_EQ(0, ranks.rank_of('c'));
    EXPECT_EQ(1, ranks.rank_of('d'));

    EXPECT_TRUE(ranks.erase('d'));

    EXPECT_EQ(1, ranks.size());
    EXPECT_EQ(-1, ranks.rank_of('a'));
    EXPECT_EQ(-1, ranks.rank_of('b'));
    EXPECT_EQ(0, ranks.rank_of('c'));
    EXPECT_EQ(-1, ranks.rank_of('d'));

    EXPECT_TRUE(ranks.erase('c'));

    EXPECT_EQ(0, ranks.size());
    EXPECT_EQ(-1, ranks.rank_of('a'));
    EXPECT_EQ(-1, ranks.rank_of('b'));
    EXPECT_EQ(-1, ranks.rank_of('c'));
    EXPECT_EQ(-1, ranks.rank_of('d'));
}