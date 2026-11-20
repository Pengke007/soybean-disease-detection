const app = getApp()

Page({
  data: { list: [] },
  onShow() {
    const lib = app.globalData.diseaseLib || {}
    const list = Object.keys(lib).map(key => ({
      key,
      name_zh: lib[key].name_zh,
      type: lib[key].type,
      symptoms: lib[key].symptoms,
      emoji: lib[key].emoji || '🌿',
      color: lib[key].color || '#1f7a4d'
    }))
    this.setData({ list })
    if (list.length === 0) {
      // 后端尚未就绪时重试加载
      app.loadDiseases();
      setTimeout(() => {
        const l2 = app.globalData.diseaseLib || {}
        this.setData({
          list: Object.keys(l2).map(key => ({
            key, name_zh: l2[key].name_zh, type: l2[key].type,
            symptoms: l2[key].symptoms, emoji: l2[key].emoji || '🌿',
            color: l2[key].color || '#1f7a4d'
          }))
        })
      }, 1500)
    }
  },
  goDetail(e) {
    wx.navigateTo({ url: '/pages/detail/detail?cls=' + e.currentTarget.dataset.cls })
  }
})
